# 50-step closed-loop gradient accuracy: A/B × slack/no-slack vs QP tolerance

Per-sample dL/d(Q,R) of a 50-step closed-loop rollout (MPC horizon 20, nx=8 nu=4, umax=1, batch=64,
seed=0) for four formulations, swept over the QP (ADMM) tolerance:
  1. A no-slack — TurboMPC hard box (OptimalControlProblem, DIRECT_CUDSS_FFI backward)
  2. A slack    — TurboMPC slack box (OptimalControlProblemSlack, gamma=1e4, Moreau penalty)
  3. B no-slack — log-barrier central-path, gamma->1e12 (pure barrier = acados tau_min), kappa=1e-6
  4. B slack    — log-barrier central-path, gamma=1e4 (barrier + Moreau slack), kappa=1e-6

Forward ADMM: rho=0.1, sigma=1e-6, alpha=1.6, adaptive rho (rho_min/max 1e-6/1e6, every 25,
tol 5), rho_f_factor=1000, max_iter=2000, check_every=1, eps_abs=eps_rel=the swept tol (turbompc.yaml
+ overrides). 1 SQP iter (linear). Script: `closed_loop_4way_sweep.py`.

**Ground truth = convergence-checked FD of each config's forward at the TIGHTEST tol (1e-9)** — the
converged "true" gradient, a fixed per-config reference for every tolerance (FD over eps in
1e-4..3e-6, per-sample plateau; non-converging samples flagged/excluded). So cos measures bias of the
analytic gradient from the converged truth, NOT AD-vs-FD self-consistency at each tol.

## Forward-backend reproducibility check (ADMM_FUSED_CUDSS vs ADMM_JAX_LOOP_CUDSS_FFI)

Same DIRECT_CUDSS_FFI backward; per-sample closed-loop gradient, batch=16:

| box | tol | cos(fused, jax-loop) | forward-cost rel-diff |
|---|---:|---:|---:|
| slack | 1e-7 | **1.000000** | 6e-7 |
| hard  | 1e-7 | **0.546** | 6.6e-7 |
| slack | 1e-1 | 0.443 | 0.33 |
| hard  | 1e-1 | 0.369 | 0.27 |

- **Slack + converged: fused == jax-loop** (cos 1.0) — the fused cuDSS kernel is validated.
- **Hard box + converged: same primal solution (cost rel-diff 6.6e-7) but DIFFERENT gradient (cos 0.55).**
  Swapping only the forward backend, with an identical converged primal, flips the hard-box gradient —
  it is so ill-conditioned it is not even backend-reproducible. The slack box is backend-stable. New
  sharp evidence that the hard-box gradient defect is intrinsic (degenerate active set / KKT), not a
  solve-accuracy artifact.
- **Loose tol: the forwards diverge** (cost rel-diff ~0.3) — fused and jax-loop stop at different
  iterates; over 50 closed-loop steps that compounds to ~30% cost difference. Loose solves are
  backend-dependent for both boxes. (Sweep uses jax-loop throughout for consistency.)

Script: `fused_check.py`.

## 4-way accuracy vs tolerance (GT = FD at tol 1e-9)

(table pending the running batch=64 sweep)

## CORRECTION: the hard-box "badness" for A was a backend bug (ADMM_JAX_LOOP_CUDSS_FFI), not the formulation

The DIRECT backward fed by the ADMM_JAX_LOOP_CUDSS_FFI **forward** returns a WRONG hard-box gradient;
the ADMM_FUSED_CUDSS forward returns the CORRECT one. Measured (backend_vs_fd.py, hard box, tol=1e-7):
cos(jax-loop, FD)=0.21, cos(fused, FD)=1.00, cos(jax-loop, fused)=0.21. Slack box: both 1.00.
(Likely cause, not yet directly inspected: the jax-loop forward's duals make the DIRECT backward read
the wrong active set; the fused kernel's duals are correct. The jax-loop hard-box forward is also
noisier — its FD GT flags 13/64 vs fused's 2/64.)

Re-run of A with **ADMM_FUSED_CUDSS** (GT = FD of the fused forward at tol 1e-9):

| tol | A no-slack JAX-loop | A no-slack FUSED | A slack FUSED |
|---:|---:|---:|---:|
| 1e-1 | 0.26 | 0.73 | 0.88 |
| 1e-3 | 0.25 | 1.0000 | 1.0000 |
| 1e-5 | 0.20 | 1.0000 | 1.0000 |
| 1e-7 | 0.22 | 1.0000 | 1.0000 |
| 1e-9 | 0.14 | 1.0000 | 1.0000 |
| GT-flagged | 13/64 | 2/64 | 5/64 |

**With the fused backend, A no-slack (hard box) reaches cos 1.0 at every tight tolerance** (>=1e-3),
1 residual sign-flip, 2 genuinely non-smooth samples. So:
- At tight tol ALL four formulations give the correct gradient (cos 1.0) -- the hard box is NOT broken.
- At loose tol all degrade (loose-solve bias from the true gradient) -- formulation-independent.
- Only ~5% of hard-box samples are genuinely non-smooth (true active-set boundaries), not 75-92%.

**Scope:** any prior "hard box bad" result using the JAX-loop backward on the HARD box is confounded
(earlier closed-loop "75-92% outliers"; the ADMM-leg cos 0.03 of the acados cross-check bd13da3). NOT
affected: slack-box results (backends agree), B/central-path, acados tau_min sweep, B-no-slack=acados.
TODO: re-verify the acados hard-box cross-check with the fused ADMM backward; revise NOTE.md.
Default backend for A switched to ADMM_FUSED_CUDSS in closed_loop_4way_sweep.py. Script: backend_vs_fd.py.
