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

# CORRECTION: acados's tau_min smoothing DOES fix the gradient (= B's log-barrier idea)

The addendum above was **wrong**. Per Frey/Diehl 2025 "Differentiable NMPC" (Eq.10, Thm.3, Fig.1,
Remark 2), acados smooths the gradient by keeping the **interior-point barrier at tau_min > 0**
(complementarity `mu_i h_i = tau_min` instead of 0). Thm.3: the solution map is then **continuously
differentiable**; Remark 2: the adjoint gives the **correct** sensitivity of the smoothed map even
when strict complementarity fails. `tau_min=0` is the exact/nonsmooth case — which is the only one I
had run. Set via `solver.options_set('tau_min', tau)` on the forward AND sensitivity solvers.

Re-run sweeping tau_min (same hard-box linear QP, 16 x0, FD of the SAME tau_min-smoothed forward):

| tau_min | FD-flagged | cos med | cos min | rel med |
|---:|---:|---:|---:|---:|
| 1e-2 | 0 | **1.0000** | 1.0000 | 2.4e-4 |
| 1e-3 | 0 | **1.0000** | 1.0000 | 1.7e-3 |
| 1e-4 | 0 | **1.0000** | 1.0000 | 4.3e-3 |
| 1e-6 | 4 | 0.9987 | 0.964 | 0.12 |
| 1e-9 | 5 | 0.655 | −0.17 | 0.99 |
| 0 (exact) | 5 | 0.624 | −0.47 | 1.00 |

**acados's tau_min smoothing fixes it**: cos=1.0 / 0-flagged for tau_min ≥ 1e-4, degrading to the
ill-defined hard gradient as tau_min→0 — exactly Fig.1 of the paper, on our problem.

## This is the same mechanism as B — a cross-validation

acados's `tau_min` (a fixed IP barrier on the inequalities) and B's log-barrier `kappa` (a relaxed
complementarity `s·y = kappa`) are the **same interior-point smoothing** of the active set. Both turn
the ill-defined hard-box gradient into a smooth, FD-consistent one. So B is validated against the
state-of-the-art acados/Diehl differentiable-MPC method.

One real difference: at very small barrier (tau_min=1e-9) acados degrades to cos 0.65, but **B stayed
cos=1.0 down to kappa=1e-11** (the kappa-sweep). Reason: B also carries the **elastic/Moreau slack**
(gamma=1e4), which provides C1 smoothing independent of the barrier — so B's robustness comes mainly
from the slack, with kappa secondary (consistent with the earlier kappa-sweep finding). acados here
uses ONLY the barrier (no slack), so it needs tau_min >~ 1e-4 to smooth. Both are valid; B's
slack+barrier is just more barrier-robust. Corrected experiment: `acados_qp_gradient.py`
(tau_min sweep), data `acados_tau_sweep.npz`.

# Addendum: qp_solver_iter_max — works in a clean solve, bypassed in the differentiable config

Tried `qp_solver_iter_max` as the real accuracy knob (vs `qp_solver_tol`, which overshoots).

- **Clean solve (no tau_min / no options_set):** the cap works and IS the fine-grained knob — the
  iter-probe gave `||x-x*||` = 9.46, 8.36, 6.77, 5.13, 3.20, 1.32, 0.11, 8e-3, 3.5e-4, 4.5e-8 for
  k=1..10. So qp_iter_max controls solve accuracy smoothly, unlike qp_solver_tol.
- **Differentiable config (tau_min set via `options_set`, two-solver sensitivity):** the build-time
  cap is **bypassed** — `get_stats('qp_iter')` returns the max and `||x-x_ref||=0` for every
  `qp_solver_iter_max` from 1 to 1000, at both tau_min=1e-4 and tau_min=0. acados solves the QP fully
  regardless. (`qp_solver_iter_max` is also build-time only; not in the runtime `options_set` list.)

## Consolidated conclusion: acados has no graceful solve-accuracy-vs-gradient tradeoff

The two solvers control gradient quality through fundamentally different knobs:

| | accuracy knob for the gradient | loosen it -> gradient |
|---|---|---|
| **our ADMM (1st-order)** | `eps_abs/eps_rel` (QP-KKT residual) | degrades **gracefully** (the RQ2 result: cos drops smoothly as eps loosens) |
| **acados HPIPM (2nd-order IPM)** | `tau_min` (the barrier/smoothing) | the QP is solved **tightly regardless**; gradient is well-defined (tau_min>=1e-4, cos=1.0) or ill-defined (tau_min->0). `qp_solver_tol` overshoots and `qp_solver_iter_max` is bypassed in the diff config. |

So for acados the meaningful axis is **tau_min (smoothing), not QP solve accuracy** — there is no
"loosen the QP solve, watch the gradient degrade" curve like ADMM has, because the IPM converges the
QP tightly in ~10 iters and the diff machinery forces a full solve. The fair cross-solver statement
is therefore: ADMM trades gradient accuracy against solve cost (eps); acados fixes the solve (tight)
and trades gradient *smoothness/bias* against the active-set fidelity (tau_min) — the two are
controlling different things.
