# acados vs turbompc vs converged-FD — independent check of the ~15% backward gap (K=1 cartpole)

**Question.** The cartpole sweep found turbompc's AD gradient `d(closed-loop cost)/d(Q,R)` matches the
convergence-verified FD in *direction* (cos≈1) but is ~16% off in *magnitude* (rel_l2≈0.16), even with
`use_full_hessian=True`. Is that a turbompc backward error, or a property of the FD/problem? Settle it
with an INDEPENDENT exact-Hessian adjoint gradient from **acados**, at the *same* OCP solution.

## Method (staged, each gated)
- **OCP replica in acados/CasADi** matching turbompc EXACTLY: cartpole RK4 DISCRETE (dt=0.04, N=25),
  external cost `J=Σ_k Q_i x_{k,i}²+R u_k²` (Q=[1,1,10,1], R=[0.1], no ½). Dynamics gate:
  CasADi RK4 == turbompc RK4 to **8.9e-16**.
- **Forward** (independent, NO warm-start): acados solves with **Gauss-Newton** (cost Hessian
  `diag(2Q,2R)`, dynamics curvature dropped) + **Levenberg-Marquardt μ·I** (= turbompc's ADMM proximal
  ρ·I) + **MERIT_BACKTRACKING** L1 line search — i.e. turbompc's exact forward recipe (turbompc's
  forward QP is GN: `D=cost.D`, no λᵀ∇²f augmentation; turbompc_solver.py:589/603). It converges
  cold to turbompc's solution: **108/128 samples match the FULL state+control trajectory to <1e-6**
  (the other 20 are different local minima — the swing-up is nonconvex). Comparison runs ONLY on the
  108 confirmed-same-solution samples.
- **Backward**: a SEPARATE acados **EXACT-Hessian** solver loads the forward iterate,
  `setup_qp_matrices_and_factorize`, `eval_adjoint_solution_sensitivity` with seed
  `s=2u0+(∂RK4/∂u0)ᵀ(2x1)` → `d(C)/d(Q,R)` (C=‖x1‖²+‖u0‖²). Canonical acados differentiable-MPC pattern.

## Result (108 confirmed same-solution samples)

| comparison | cos median (min) | relmag ‖a‖/‖b‖ median (IQR) |
|---|---|---|
| **acados-exact vs converged-FD** | +1.0000 (+1.0000) | **1.0000** (1.000–1.000) |
| **turbompc-AD vs converged-FD** | +1.0000 (+0.9996) | **0.8373** (0.833–0.841) |
| acados-exact vs turbompc-AD | +1.0000 (+0.9996) | 1.1943 (1.189–1.200) |

## Verdict
**The ~16% gap is a turbompc backward-pass magnitude error.** Two independent references — acados's
exact-Hessian parametric adjoint and the convergence-verified FD — agree to **relmag=1.0000** (exact
in direction AND magnitude). turbompc's AD has the correct direction (cos=1.0000) but is **0.837× the
true magnitude** (≈16% too small). `|0.837−1|=0.163` exactly reproduces the sweep's `rel_l2≈0.16`.

So for the diffmpc-as-policy gradient: **direction is reliable; turbompc's magnitude is systematically
biased low by ~16%** (cartpole). Cause is open — it is NOT a Gauss-Newton/Hessian-approximation issue
(full Lagrangian Hessian is used in the backward); candidates are the implicit-diff KKT linear solve /
a scaling in the adjoint. For magnitude-insensitive optimizers (Adam) this is harmless; for raw-scale
gradient use it is a real bias.

_Reproduce:_ acados side (in the `turbompc-acados` Docker, mounting the workspace) —
`python acados_cartpole_gradient.py --check_dynamics|--check_forward|--gradient`; local 3-way —
`python compare_acados_turbompc.py`.
