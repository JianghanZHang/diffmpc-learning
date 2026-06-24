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

# Addendum: acados QP-level gradient — the hard-box gradient is ill-defined (formulation, not solver)

Tried to run the same gradient-accuracy check on acados's HPIPM (sweep `qp_solver_iter_max`,
exact-Hessian adjoint vs convergence-checked FD). Two findings:

1. **Iter-cap conflicts with the sensitivity solver.** With `with_solution_sens_wrt_params=True`
   (+ `qp_solver_ric_alg=0`), acados forces a converged forward solve, so `qp_solver_iter_max` is
   ignored (rel_sol_err ≈ 8e-17 for all k). A genuine solve-accuracy sweep needs the cartpole's
   **two-solver** pattern (a capped forward solver + a separate sensitivity solver loading its
   iterate). Not yet wired up.

2. **The hard-box gradient is ill-defined here — three "exact" estimates disagree** (16 random x0,
   x0~N(0,5²) so the control box binds heavily; 10 non-FD-flagged samples):

   | comparison | median cos |
   |---|---:|
   | acados exact-adjoint vs TurboMPC hard-box backward | 0.32 |
   | acados exact-adjoint vs FD | 0.61 |
   | TurboMPC hard-box backward vs FD | 0.03 |

   None agree. The hard-box QP solution sits on degenerate active sets (saturated controls with
   ~0 multipliers, strict-complementarity failure), where `dx*/d(Q,R)` is genuinely discontinuous —
   so acados's adjoint, TurboMPC's backward, and FD each give a different (one-sided / straddled)
   answer. **Even acados's gold-standard exact-Hessian IPM adjoint cannot make the hard-box gradient
   well-defined** — it's a property of the formulation, not the solver. This is the open-loop,
   single-QP analogue of the closed-loop hard-box pathology, and it explains why the soft/slack
   formulations (A = TurboMPC `use_slack=True`, B = log-barrier) give cos=1.0 vs FD while the hard box
   does not: the slack/barrier regularizes the degenerate active set into a smooth, well-defined map.

**To get a meaningful acados gradient-vs-accuracy curve, acados must solve a SOFT box** (acados
supports L2 slacks) so the gradient is well-defined; then the iter-cap (two-solver) sweep is
informative. Data: `acados_qp_itersweep.npz` (gAD = acados adjoint per sample).
