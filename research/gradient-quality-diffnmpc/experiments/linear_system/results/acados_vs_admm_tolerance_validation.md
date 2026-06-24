# Validation: is acados-HPIPM's QP-tolerance equivalent to our ADMM's? (NO)

**Run 2026-06-24, turbompc-acados Docker.** Same hard-box linear-MPC QP (nx=8, nu=4, H=20, umax=1),
same x0. Question: at the same set QP-tolerance X, do acados HPIPM (`qp_solver_tol_*`) and our
TurboMPC ADMM (`eps_abs=eps_rel`) solve the QP to the equivalent accuracy?

Solution error vs the tight reference solution (`‖x − x*‖∞`):

| set tol | ADMM ‖x−x*‖ | ADMM iters | HPIPM ‖x−x*‖ | HPIPM qp_iter |
|---:|---:|---:|---:|---:|
| 1e-1 | 3.30   | 28  | 2.2e-2  | 8  |
| 1e-2 | 0.50   | 49  | 2.4e-3  | 8  |
| 1e-3 | 0.056  | 80  | 1.2e-5  | 9  |
| 1e-4 | 6.8e-3 | 120 | 1.2e-5  | 10 |
| 1e-6 | 6.6e-5 | 205 | 1.6e-12 | 10 |
| 1e-9 | 6.6e-8 | 332 | 0       | 11 |

acados HPIPM also reports its achieved KKT residual: set 1e-1→5.2e-3, 1e-4→2.1e-8, 1e-9→8.5e-14
(always ≤ set, often 4+ orders tighter).

## Finding: NOT equivalent

- **ADMM (first-order, linear convergence):** `‖x−x*‖ ≈ eps`. The solution accuracy *tracks* the set
  tolerance; iters grow smoothly with log(1/eps).
- **HPIPM (second-order interior point, quadratic convergence):** `‖x−x*‖ ≪ qp_tol` — it *overshoots*
  the set tolerance by orders of magnitude; qp_iter steps discretely (8→11) across the whole range.
  `qp_solver_tol` is a ceiling HPIPM blows past, not a level it stops at.

(Confirmed `nlp_qp_tol_strategy = 'FIXED_QP_TOL'` is the acados default, so the QP tol I set is not
adaptively re-tightened — the overshoot is intrinsic to IPM convergence, not a setting artifact.)

## Implication for a fair cross-solver comparison

Comparing gradient accuracy vs the **set** tolerance is invalid (HPIPM would look artificially exact at
any "tolerance"). Use the **achieved** QP-KKT residual (or `‖x−x*‖`) as the common axis. To obtain
genuinely loose HPIPM solves for the curve, cap **`qp_solver_iter_max`** (the discrete 8→11 steps show
iteration count is the real knob), not `qp_solver_tol`.
