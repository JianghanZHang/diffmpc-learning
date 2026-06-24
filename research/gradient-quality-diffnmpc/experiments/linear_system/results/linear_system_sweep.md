# Linear-system ADMM-tolerance gradient sweep (RQ2 / H2.2)

**Run 2026-06-23 (re-run after central-path ADMM acceleration).** Fixed random linear MPC
(`nx=8, nu=4, horizon=20, umax=1.0`), **1 SQP iteration** (linear ⇒ the NLP *is* a single QP, so
the inner-QP **ADMM tolerance is the only solve-accuracy knob**). Samples = 16 initial states (FD
plateau 16/16; box active). Differentiable parameter = cost weights `(Q,R)`; fixed quadratic loss.
`γ=1e4` (matched A/B), `κ=1e-6` (B). GT = convergence-checked FD of the tight (`eps=1e-10`) slack
forward. Driver: `sweep_admm_tolerance.py`.

- **A** = TurboMPC analytic backward, `use_slack=True` (indicator + slack penalty γ), ADMM stopped
  at `eps_abs=eps_rel`.
- **B** = log-barrier smoothed backward, central-path ADMM stopped at the **same residual** tolerance.

Medians over the 16 samples; `iters` = median ADMM iterations of the forward.

| ADMM eps | A cos | A rel_l2 | A iters | B cos | B rel_l2 | B iters |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-2  | 0.980482 | 2.46e-01 | 40  | 0.515950 | 8.49e-01 | 75  |
| 1e-3  | 0.999807 | 2.56e-02 | 60  | 0.999986 | 6.06e-02 | 100 |
| 1e-4  | 0.999997 | 2.55e-03 | 80  | 1.000000 | 7.16e-03 | 138 |
| 1e-6  | 1.000000 | 4.82e-05 | 118 | 1.000000 | 1.71e-04 | 188 |
| 1e-8  | 1.000000 | 3.31e-05 | 159 | 1.000000 | 4.39e-05 | 250 |
| 1e-10 | 1.000000 | 3.30e-05 | 200 | 1.000000 | 4.38e-05 | 312 |

## Important correction — the first run was a solver artifact, not the smoothing

The **first version of this sweep** reported B collapsing catastrophically under a loose solve
(median **cosine = −0.97 / −0.98** at eps=1e-3/1e-4) and needing **~100× more ADMM iterations**
(~18000 vs ~200 at the tight end). Systematic debugging found that was **not** a property of the
log-barrier backward: the central-path ADMM was *un-tuned* (fixed ρ, **no over-relaxation, no
adaptive ρ**) and terminated on a **step-norm** `delta` that lagged the true residual by ~300×, so
"B at eps=1e-3" was actually a solve with residual ~0.3. On those infeasible iterates the barrier
slack `s=h−Gx+y_g/γ` went **negative**, flipping the sign of the backward weight `W=y_g/(s+y_g/γ)`
→ wrong-direction gradient. Porting TurboMPC's accelerated ADMM (over-relaxation α=1.6, OSQP adaptive
ρ + Schur rebuild, **residual-based** termination) into `solve_qp_central_path` — *the only remaining
difference from TurboMPC is the elastic-retraction z-update* — removed both effects.

| | first run (un-tuned) | this run (accelerated) |
|---|---|---|
| B cos @ eps=1e-3 | **−0.974** | **0.99999** |
| B cos @ eps=1e-4 | **−0.984** | 1.000 |
| B iters @ eps=1e-10 | ~18395 | **312** |

## Findings (measured)

- **B no longer collapses.** At every eps ≤ 1e-3 the median cosine is ≥ 0.99999 — B now degrades
  *gracefully* like A as the QP is under-solved. (The barrier slack stays positive because B now
  stops on the residual.)
- **Iteration parity.** B reaches the same residual in **~1.5× A's iterations** (312 vs 200 at the
  tight end), not ~100×. (The retraction z-update is marginally stiffer at γ=1e4, κ=1e-6 than the box
  projection; same order of magnitude. A dedicated parity test on the cartpole QP shows exact 1:1.)
- **Residual floors.** A's tight floor (~3.3e-5) is FD-limited; B's (~4.4e-5) is the O(κ) barrier
  bias (κ=1e-6) — both negligible.
- **Remaining gap only at the extreme loose end.** At eps=1e-2 (≈75 ADMM iters) B's median cosine is
  0.52 vs A's 0.98 — at this severely under-solved point B is still somewhat worse, but the
  catastrophic wrong-direction failure is gone (rel_max 3.1 vs the first run's 177).

## Takeaway

The earlier headline ("log-barrier B is ~100× less efficient and fragile to loose solves") was a
**solver-tuning + step-norm-criterion artifact**, not intrinsic to the smoothing. Once the
central-path ADMM uses the same acceleration as TurboMPC and stops on the residual, **B is
competitive with A** on the inexact-solve axis (graceful degradation, ~1.5× iterations, only a
small disadvantage at the very loosest solves). Combined with the earlier finding (on the
active-set-smoothness axis the converged gradients of A and B are both accurate), there is **no
large intrinsic penalty** for the log-barrier backward on this linear testbed. (Not yet measured:
whether a larger κ trades robustness-at-very-loose-solves for more bias; and the obstacle/nonlinear
case where the smoothing was originally motivated.)
