# Quadrotor — gradient accuracy vs NLP-KKT tolerance × closed-loop coupling K

**Setup.** Quadrotor (13 states [pos,vel,quat,omega], nu=4 [thrust,torques]), hover reference, perturbations about hover, N=25, dt=0.04, RK4. No inequality constraints (umax=1e7). Same diffmpc-as-policy gradient ∂(K-step closed-loop cost)/∂(Q,R weights), 128 samples, convergence-verified FD GT at gt_tol=1e-9. Forward `admm_fused_cudss`, backward `direct_cudss_ffi`.

> **Post-fix — backward dynamics-Hessian now follows the configured integrator** (branch `backward-gradient-fix`, commit `cccba81`). `get_dynamics_lagrangian_hessian` differentiates the actual RK4 map `λᵀ∇²(predict_next_state)` instead of the hardcoded Euler `dt·λᵀ∇²f`. At every converged tolerance the AD↔FD **magnitude** gap collapses: `rel_l2` median **~0.05 → ~5e-4**. The fix **also resolves the two K=1 direction-mismatches** seen pre-fix (the largest-gradient samples, where the ~5% magnitude error had tipped into a direction error): `cos_min` 0.984 → 0.9999999, `#cos<0.99` 2 → 0. Residual error at `tol = 1e0 / 1e1` is *forward* under-convergence, not the backward. **Pre-fix** `rel_l2` median at tol=1e-9 was K=1/5/20 = 6.20e-2 / 5.29e-2 / 4.89e-2 (cos_min 0.984 at K=1).

_Columns:_ `cos` = per-sample cosine(AD, converged-FD GT) over non-flagged samples; `rel_l2` = per-sample ‖g_AD−g_GT‖/‖g_GT‖ (captures magnitude error that cos misses); `achieved KKT` = median first-order residual the forward reached; `reached`/`overshoot` = fraction with achieved ≤ tol / < tol/10. All runs: 0 samples flagged.

## K = 1

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | 0.9915 | +0.601 | 58 | 8.04e-01 | 9.95e-01 | 7.1e+00 | 1.00 | 0.02 |
| 1e+00 | 1.0000 | +0.998 | 0 | 5.64e-03 | 2.27e-01 | 5.2e-01 | 1.00 | 0.06 |
| 1e-01 | 1.0000 | +1.000 | 0 | 9.91e-04 | 4.93e-02 | 5.1e-02 | 1.00 | 0.04 |
| 1e-03 | 1.0000 | +1.000 | 0 | 6.12e-04 | 2.51e-03 | 5.5e-04 | 1.00 | 0.02 |
| 1e-05 | 1.0000 | +1.000 | 0 | 7.03e-04 | 2.11e-03 | 3.2e-06 | 1.00 | 0.24 |
| 1e-09 | 1.0000 | +1.000 | 0 | 5.86e-04 | 2.66e-03 | 9.0e-10 | 1.00 | 0.05 |

## K = 5

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | 0.9751 | -0.363 | 71 | 8.40e-01 | 1.14e+01 | 7.1e+00 | 1.00 | 0.02 |
| 1e+00 | 1.0000 | +0.998 | 0 | 6.81e-03 | 2.81e-01 | 5.2e-01 | 1.00 | 0.06 |
| 1e-01 | 1.0000 | +1.000 | 0 | 9.96e-04 | 4.63e-02 | 5.1e-02 | 1.00 | 0.04 |
| 1e-03 | 1.0000 | +1.000 | 0 | 4.57e-04 | 2.27e-03 | 5.5e-04 | 1.00 | 0.02 |
| 1e-05 | 1.0000 | +1.000 | 0 | 5.09e-04 | 2.30e-03 | 4.2e-06 | 1.00 | 0.19 |
| 1e-09 | 1.0000 | +1.000 | 0 | 4.94e-04 | 1.59e-03 | 9.4e-10 | 1.00 | 0.02 |

## K = 20

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | -0.0055 | -0.978 | 93 | 1.02e+00 | 2.25e+04 | 7.1e+00 | 1.00 | 0.02 |
| 1e+00 | 1.0000 | +0.997 | 0 | 7.36e-03 | 3.04e-01 | 5.2e-01 | 1.00 | 0.06 |
| 1e-01 | 1.0000 | +1.000 | 0 | 1.18e-03 | 4.55e-02 | 5.1e-02 | 1.00 | 0.04 |
| 1e-03 | 1.0000 | +1.000 | 0 | 5.04e-04 | 2.03e-03 | 5.5e-04 | 1.00 | 0.04 |
| 1e-05 | 1.0000 | +1.000 | 0 | 4.20e-04 | 2.05e-03 | 4.3e-06 | 1.00 | 0.20 |
| 1e-09 | 1.0000 | +1.000 | 0 | 4.43e-04 | 2.30e-03 | 9.5e-10 | 1.00 | 0.02 |
