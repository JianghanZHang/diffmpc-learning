# Linear-system ADMM-tolerance gradient sweep (RQ2 / H2.2)

**Run 2026-06-23.** Fixed random linear MPC (`nx=8, nu=4, horizon=20, umax=1.0`), **1 SQP
iteration** (linear ⇒ the NLP *is* a single QP, so the inner-QP **ADMM tolerance is the only
solve-accuracy knob**). Samples = 16 initial states (FD plateau 16/16; `max|u|∈[1.00,1.03]` so
the box is active). Differentiable parameter = cost weights `(Q diag, R diag)`; fixed quadratic
loss. `γ=1e4` (matched A/B), `κ=1e-6` (B). Driver: `sweep_admm_tolerance.py`.

- **A** = TurboMPC analytic backward, `use_slack=True` (indicator + quadratic slack penalty γ).
  ADMM stopped at `eps_abs=eps_rel` (residual criterion).
- **B** = log-barrier smoothed backward (`central_path_nlp_grad`, fixed κ). Central-path ADMM
  stopped at `cp_tol` (step-norm criterion).
- **GT** = convergence-checked FD of the **tight** (`eps=1e-10`) slack forward (the true gradient).

Medians over the 16 (all non-flagged) samples; `iters` = median ADMM iterations of the forward.

| ADMM eps | A cos | A rel_l2 | A iters | B cos | B rel_l2 | B iters |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-2  | 0.980482 | 2.46e-01 | 40  | 0.695283 | 4.29e+01 | 102 |
| 1e-3  | 0.999807 | 2.56e-02 | 60  | **-0.973839** | 3.35e+01 | 880 |
| 1e-4  | 0.999997 | 2.55e-03 | 80  | **-0.983815** | 4.30e+00 | 2854 |
| 1e-6  | 1.000000 | 4.82e-05 | 118 | 0.999990 | 4.02e-02 | 8043 |
| 1e-8  | 1.000000 | 3.31e-05 | 159 | 1.000000 | 4.11e-04 | 13173 |
| 1e-10 | 1.000000 | 3.30e-05 | 200 | 1.000000 | 6.72e-05 | 18395 |

(`summary.csv` also has the per-tolerance `rel_max` over samples.)

## Findings (measured)

- **A (indicator+slack) degrades gracefully and monotonically.** `cos ≥ 0.98` at *every* ADMM
  tolerance (direction always usable); `rel_l2` falls ~linearly with eps (2.5e-1 → 2.6e-2 → 2.6e-3
  → … → 3.3e-5). A reaches a usable gradient (`rel_l2 < 1e-2`, `cos > 0.999`) in **~60 ADMM
  iterations**; its tight floor (~3.3e-5) is FD-limited, not bias.
- **B (log-barrier) is fragile to an under-solved QP — and can point the wrong way.** At the
  intermediate tolerances `eps = 1e-3, 1e-4` the median **cosine is negative** (−0.97, −0.98): the
  gradient points *opposite* the truth. B recovers a correct gradient only at `eps ≤ 1e-6`, and
  needs **~8000–18000 ADMM iterations** (vs A's ~120–200) to match A's accuracy. B's tight floor
  (~6.7e-5) is the O(κ) barrier bias.
- **Iteration-efficiency gap (the honest, criterion-independent axis).** For a usable-direction
  gradient (`cos > 0.99`): A needs ~60 iters, B needs ~8000 — **~100×**. (A's nominal `eps`
  (residual) and B's `cp_tol` (step-norm) are *not* the same criterion, so compare on achieved
  iterations, not nominal tolerance.)

## Interpretation (hypothesis — mechanism not directly measured)

Likely cause of B's fragility: B folds inequalities into the Hessian via `W = y_g/(s + y_g/γ)`,
`s = h − Gx + y_g/γ`. At a loosely-converged central-path solve the primal `Gx` and dual `y_g` are
far from the fixed point `s·y_g = κ`, so `s` and `W` are badly estimated (and can take the wrong
sign), corrupting the `GᵀWG` Hessian augmentation — hence a wrong-magnitude or wrong-direction
gradient. A's hard active-set mask is binary and is identified correctly even from a loose solve,
so A suffers only the (graceful) linearization-point error. **Unverified** — confirm by logging
`W`, `s`, and the prim/dual residual at the loose-solve point.

## Takeaway for the project

For the **RQ2 "cheaper solve"** goal, the log-barrier smoothing (built for RQ1 smoothness across
active-set changes) carries a **large hidden cost**: it demands a *much* tighter inner-QP solve
(~100× the ADMM iterations) than TurboMPC's indicator+slack backward to yield a usable gradient,
and at loose/intermediate solves it can return a wrong-direction gradient. On this linear testbed
**A dominates B on the inexact-solve axis** — the opposite trade-off from the active-set-smoothness
axis. Not yet measured: whether a larger κ buys B robustness to loose solves (at the cost of more
bias), and whether a better-tuned central-path ADMM (adaptive ρ) closes the iteration gap.
