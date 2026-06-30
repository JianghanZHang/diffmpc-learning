# Cartpole — gradient accuracy vs NLP-KKT tolerance × closed-loop coupling K

**Setup.** Cartpole swing-up (pole-down x0, nx=4, nu=1), N=25, dt=0.04, RK4. No inequality constraints (umax=1e7). diffmpc-as-policy gradient ∂(K-step closed-loop cost)/∂(Q,R weights). Per-sample metric: each of 128 initial states (2 seeds × 64) = one sample. Ground truth = convergence-verified per-sample FD at gt_tol=1e-9 (FD step shrunk over [1e-5,1e-6,3e-7,1e-7] until the gradient plateaus). Forward `admm_fused_cudss`, backward `direct_cudss_ffi`.

> **Post-fix — backward dynamics-Hessian now follows the configured integrator** (branch `backward-gradient-fix`, commit `cccba81`). `get_dynamics_lagrangian_hessian` differentiates the actual RK4 map `λᵀ∇²(predict_next_state)` instead of the hardcoded Euler `dt·λᵀ∇²f`. The AD↔FD **magnitude** gap collapses at every converged tolerance: `rel_l2` median **~0.16 → ~5e-5** (the old flat ~16% "backward floor" at all tol ≤ 1e-1 is gone). Direction (`cos`) was already ~1.0 and is unchanged. Residual error at `tol = 1e0 / 1e1` is *forward* under-convergence (differentiating a non-fixed-point), not the backward. **Pre-fix** `rel_l2` median at tol=1e-9 was K=1/5/20 = 1.63e-1 / 1.77e-1 / 9.97e-2.

_Columns:_ `cos` = per-sample cosine(AD, converged-FD GT) over non-flagged samples; `rel_l2` = per-sample ‖g_AD−g_GT‖/‖g_GT‖ (captures magnitude error that cos misses); `achieved KKT` = median first-order residual the forward reached; `reached`/`overshoot` = fraction with achieved ≤ tol / < tol/10. Flagged (no FD plateau → at a discontinuity, excluded from cos): 0 at K=1,5; 1 sample at K=20.

## K = 1

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | 0.9984 | -0.999 | 45 | 8.82e-01 | 2.00e+01 | 8.5e+00 | 1.00 | 0.00 |
| 1e+00 | 1.0000 | -0.994 | 2 | 7.67e-03 | 1.36e+01 | 7.7e-01 | 1.00 | 0.00 |
| 1e-01 | 1.0000 | +1.000 | 0 | 7.38e-04 | 5.78e-01 | 7.8e-02 | 1.00 | 0.00 |
| 1e-03 | 1.0000 | +1.000 | 0 | 7.95e-05 | 6.61e-03 | 7.5e-04 | 1.00 | 0.01 |
| 1e-05 | 1.0000 | +1.000 | 0 | 8.30e-05 | 8.29e-04 | 2.7e-06 | 1.00 | 0.05 |
| 1e-09 | 1.0000 | +1.000 | 0 | 8.76e-05 | 8.44e-04 | 7.8e-10 | 1.00 | 0.00 |

## K = 5

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | 0.9998 | -1.000 | 17 | 3.02e-01 | 5.16e+01 | 8.5e+00 | 1.00 | 0.00 |
| 1e+00 | 1.0000 | +1.000 | 0 | 3.63e-03 | 1.26e+00 | 7.7e-01 | 1.00 | 0.00 |
| 1e-01 | 1.0000 | +1.000 | 0 | 3.09e-04 | 2.14e-02 | 7.8e-02 | 1.00 | 0.00 |
| 1e-03 | 1.0000 | +1.000 | 0 | 4.05e-05 | 1.96e-04 | 7.5e-04 | 1.00 | 0.00 |
| 1e-05 | 1.0000 | +1.000 | 0 | 4.61e-05 | 2.37e-04 | 2.7e-06 | 1.00 | 0.04 |
| 1e-09 | 1.0000 | +1.000 | 0 | 4.51e-05 | 1.83e-04 | 7.6e-10 | 1.00 | 0.00 |

## K = 20

| set tol | cos median | cos min | #cos<0.99 | rel_l2 median | rel_l2 max | achieved KKT | reached | overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+01 | 0.9992 | -0.992 | 18 | 6.23e-01 | 3.71e+01 | 8.5e+00 | 1.00 | 0.00 |
| 1e+00 | 1.0000 | +0.998 | 0 | 1.55e-02 | 1.51e+00 | 7.7e-01 | 1.00 | 0.00 |
| 1e-01 | 1.0000 | +1.000 | 0 | 5.68e-04 | 4.85e-01 | 7.8e-02 | 1.00 | 0.00 |
| 1e-03 | 1.0000 | +1.000 | 0 | 3.61e-05 | 4.42e-02 | 7.5e-04 | 1.00 | 0.00 |
| 1e-05 | 1.0000 | +1.000 | 0 | 3.28e-05 | 1.04e-02 | 2.3e-06 | 1.00 | 0.08 |
| 1e-09 | 1.0000 | +1.000 | 0 | 3.64e-05 | 1.03e-02 | 7.8e-10 | 1.00 | 0.00 |
