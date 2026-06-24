# Quadrotor closed-loop (nu=4 nonlinear): tractable; a noise-floor lesson; weak mechanism so far

**Run 2026-06-24.** Closed-loop hard-box-vs-slack gradient study on the **nu=4 nonlinear quadrotor**
(13 states, RK4, hover regulation, control box). Driver: `closed_loop_quadrotor.py`.

## Two corrections (this experiment had two false starts — both methodological)

1. **"Intractable" was wrong — GPU contention.** An early claim that the rollout-grad doesn't compile
   was a leftover-process artifact. Clean runs: sqp=10 n=2 ≈120 s; full sweep AD ≈82 s. The FD is just
   *slow* (~33–40 min/run), not intractable. See [[isolate-gpu-contention-before-boundary]].
2. **A "cos=0.616 outlier" was wrong — FD below the noise floor.** Diagnosed (2026-06-24): the rollout
   cost is non-deterministic at **~8e-8 relative** (cuDSS GPU non-determinism over 50 chained 13-state
   solves) — ~100× the cartpole's floor. The original FD used eps **3e-5…1e-6**, *below* that floor, so
   per-weight FD was noise-dominated and manufactured a spurious outlier (a sample that read cos=0.616
   at small eps reads **cos=0.99975** with eps=1e-3). **Lesson: the convergence-checked FD's usable-eps
   window has a system-dependent lower bound (the cost noise floor); the 13-state quadrotor needs
   eps≈1e-3, not the cartpole's 1e-5.** The FD eps_seq is now `(1e-3, 3e-4, 1e-4)` for the quadrotor.

## Results (noise-floor-corrected FD, eps 1e-3..1e-4, sqp=8, n=6)

| box | backward | cos median | cos min | #cos<0.99 | max rel_l2 |
|---|---|---:|---:|---:|---:|
| umax=1.2 (mild) | hardbox | 1.00000 | 0.976 | — | — |
|  | slack | 1.00000 | 0.964 | — | — |
| umax=1.05 (tight) | hardbox | 0.99998 | **0.9890** | 1/6 | 1.6e-1 |
|  | slack | 1.00000 | **0.99999** | 0/6 | 5e-3 |
| umax=1.0 (very tight) | hardbox | *(pending)* | | | |
|  | slack | *(pending)* | | | |

**So far the mechanism is present but WEAK.** At umax=1.05 the slack backward is uniformly
FD-consistent (cos ≥ 0.99999, rel ≤ 5e-3) while the hard box deviates on one sample (cos 0.989, rel
0.16). The direction matches the linear/cartpole finding (slack ≥ hardbox in FD-consistency), but the
quadrotor box at umax=1.05 (≈1.07× hover) barely engages — the dramatic pathology (negative cos) seen
on the linear nu=4 / cartpole tight box does NOT appear here. A **very tight** box (umax=1.0, ≈1.02×
hover, thrust saturates immediately) is running to see whether the signal strengthens (or the recovery
becomes infeasible). Data: `closed_loop_quadrotor_*_umax{1p2,1p05_bigeps,1p0}.npz`.

The clean confirmation of the mechanism stands on the linear (nu=4) and cartpole (nu=1 nonlinear)
systems. The quadrotor's contribution so far: (a) the approach scales to 13 states; (b) a real
methodological caveat — the closed-loop FD noise floor is system-size-dependent and must be cleared.
