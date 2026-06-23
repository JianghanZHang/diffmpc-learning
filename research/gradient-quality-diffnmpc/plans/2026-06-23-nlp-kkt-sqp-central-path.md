# NLP-KKT convergence test — SQP with the central-path inner QP solver

**Goal:** Show the central-path ADMM, used as the **inner QP solver in an SQP outer loop** on the nonlinear cartpole, drives the **NLP-KKT residual** (TurboMPC's `convergence_error`) below tolerance, for **both** the hard-box NLP (matches `TurboMPCSolver.solve` with no slack) and the soft-box NLP (matches `use_slack=True` at the same `γ`).

Work from `/home/jianghan/Workspace/diffmpc-learning` on branch `central-path-admm`. GPU + cuDSS. Package is `src/diffmpc_learning/` (src-layout); tests under `tests/` (conftest adds `src`, `diffmpc2`, and `research/gradient-quality-diffnmpc/experiments/cartpole` to `sys.path`).

## Files
- Modify: `src/diffmpc_learning/solvers/central_path_admm.py` — extend `solve_qp_central_path` to also return the converged duals.
- Create: `src/diffmpc_learning/solvers/sqp.py` — `sqp_central_path(...)`, the SQP outer loop.
- Create: `tests/python/solvers/test_sqp_central_path.py` — the two NLP-KKT tests.

## TurboMPC mechanics to reuse (verified, file:line in diffmpc2)
- `TurboMPCSolver._build_qp_data(states, controls, problem_params) -> QPData` (`turbompc_solver.py:1101`) — linearizes the OCP into a QP at the current iterate.
- `TurboMPCSolver._compute_first_order_convergence_error(qp_data, states, controls, slacks, admm_state, problem_params) -> (conv, eq_err, ineq_err, ineq_vals, l, u)` (`turbompc_solver.py:1161`). `conv = max(stationarity, eq, ineq, slack_stationarity)` where stationarity `= _apply_P(qp,x)+q+_apply_Ct(qp,y_f_0,y_f_dyn)+_apply_Gt(qp,y_g)`. It reads `admm_state.y_f_0, .y_f_dyn, .y_g` and `slacks`. THIS is the NLP-KKT residual.
- `backtracking_linesearch(program, problem_params, solver_params, states, controls, slacks, states_new, controls_new, slacks_new) -> ((states,controls,slacks), alpha)` (`turbompc/solvers/linesearch.py:54`). Treats `*_new` as the full candidate; interpolates `cur + alpha*(new-cur)`. Use it for globalization.
- `ADMMState(x_blocks, y_g, y_f_0, y_f_dyn, z_g, xi_g, rho_bar)` (`admm/admm.py:27`) — container `_compute_first_order_convergence_error` expects.
- Reference: `TurboMPCSolver.solve(initial_guess, problem_params, weights={}) -> TurboMPCSolution` with `.states, .controls, .convergence_error, .num_iter`. Build the solver/OCP via the cartpole benchmark's `build_cartpole_problem(horizon, umax, dt)` + `OptimalControlProblem` + `TurboMPCSolver(... ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI)` (see `tests/python/solvers/test_central_path_admm.py` for the exact construction + `load_solver_params`).

## Step 1 — extend `solve_qp_central_path` to return duals
Currently returns `(x_blocks, info)`. Change to `(x_blocks, duals, info)` where `duals = (y_f_0, y_f_dyn, y_g)` are the converged ADMM duals from the while-loop state (already tracked). `y_g` has the **stacked** one-sided shape `(N+1, 2m)`. Keep `info` keys unchanged. Update the existing 7 tests in `test_central_path_admm.py` to unpack the new return (`x, _, info = solve_qp_central_path(...)`), changing nothing else. Re-run them — still 7/7.

## Step 2 — `sqp_central_path` (new module `sqp.py`)
Signature:
```python
def sqp_central_path(solver, problem_params, *, slack_weight, target_kappa,
                     max_sqp_iter=60, tol=1e-5, linesearch=True, weights=None,
                     rho_bar=0.1, cp_max_iter=50000, cp_tol=1e-11) -> dict
```
`solver` is a constructed `TurboMPCSolver` (so you can call `solver._build_qp_data`, `solver._compute_first_order_convergence_error`, `solver.program`, `solver.params`). Loop, per SQP iteration:
1. `qp = solver._build_qp_data(states, controls, problem_params)` (the ORIGINAL two-sided QP).
2. `qp1 = to_one_sided(qp, slack_weight)`; build a cuDSS Schur solver once (`make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params)`).
3. `x_new, (y_f_0, y_f_dyn, y_g_stacked), cp_info = solve_qp_central_path(qp1, schur, target_kappa=target_kappa, slack_weight=slack_weight, rho_bar=rho_bar, max_iter=cp_max_iter, tol=cp_tol)`.
4. Unpack `states_new, controls_new` from `x_new` (`x_new[:, :nx]`, `x_new[:, nx:]`).
5. **Map stacked duals → original two-sided multiplier**: `m = qp.ineq.G.shape[1]`; `y_g = y_g_stacked[:, :m] - y_g_stacked[:, m:]` (upper minus lower). (Rationale: `[G;−G]ᵀ[ν_u;ν_l] = Gᵀ(ν_u−ν_l)`.)
6. **slacks**: for the hard case use zeros `(N+1, m)`; for the soft case, the original-constraint slack — recover it as the box violation of `states_new/controls_new` against `qp.ineq.l/u`, or carry `xi` from the solve mapped to the original constraint. (Simplest consistent choice: `slacks = max(Gx−u,0) − max(l−Gx,0)` evaluated at the new iterate; document the choice.)
7. **Globalize**: if `linesearch`, `(states, controls, slacks), alpha = backtracking_linesearch(solver.program, problem_params, solver.params, states, controls, slacks, states_new, controls_new, slacks_new)`; else full step.
8. **NLP-KKT residual**: build `admm_state = ADMMState(x_blocks=pack(states,controls), y_g=y_g, y_f_0=y_f_0, y_f_dyn=y_f_dyn, z_g=zeros, xi_g=slacks, rho_bar=rho_bar)`; `conv, *_ = solver._compute_first_order_convergence_error(qp_at_new, states, controls, slacks, admm_state, problem_params)` where `qp_at_new = solver._build_qp_data(states, controls, problem_params)`. Append `conv` to a history list.
9. Stop when `conv < tol` or `max_sqp_iter` reached.

Return `{"states":…, "controls":…, "conv_history": jnp.array([...]), "num_iter":…, "final_conv":…}`. (A Python while loop over SQP iterations is fine — it need not be jitted; the inner cuDSS solve is the cost.)

## Step 3 — tests (`test_sqp_central_path.py`)
Reuse the cartpole construction from `test_central_path_admm.py` (`build_cartpole_problem`, pole-down `[0,0,π,0]`, `umax=2.0`, the TurboMPCSolver build with cuDSS backends, `load_solver_params`).

**`test_nlp_kkt_converges_hard_box`**: build solver with NO slack; `ref = solver.solve(solver.initial_guess(pp), pp)`. Run `sqp_central_path(solver, pp, slack_weight=1e8, target_kappa=1e-7, linesearch=True)`. Assert: `conv_history` is decreasing overall (final ≤ first/10 at least) and `final_conv < 1e-4`; final states/controls match `ref.states/controls` to rel-ℓ∞ `< 1e-2`. (Both solve the hard NLP.)

**`test_nlp_kkt_converges_soft_box`**: build solver/QP with `use_slack=True`, `slack_penalization_weight=γ=1e2`; `ref = solver.solve(...)` (soft). Run `sqp_central_path(..., slack_weight=1e2, target_kappa=1e-7)`. Assert: `final_conv < 1e-4` and final solution ≈ the soft `ref` to rel-ℓ∞ `< 1e-2`.

## Convergence guidance (this is the risky part — iterate, don't fake)
- A fixed `κ>0` floors the NLP-KKT residual at ≈`O(κ)`; `κ=1e-7` should clear `1e-4`. If `final_conv` plateaus above `1e-4`, FIRST lower `target_kappa` (1e-8) and/or raise `cp_max_iter`; a short κ-anneal (e.g. start 1e-3, ×0.1 per SQP iter down to 1e-8) is allowed if a fixed κ won't clear tol — implement it as an option and use it.
- The pole-down swing-up is highly nonlinear; if full steps diverge, line search (reused) should fix it. If it still won't converge, raise `max_sqp_iter`.
- **Do NOT weaken the assertions to force a pass.** If after the sanctioned tuning (κ, anneal, max_iter, line search) the NLP-KKT residual genuinely won't reach `1e-4` or the solution won't match the reference, STOP and report **DONE_WITH_CONCERNS** with the actual `conv_history` arrays and the rel-error vs the reference for BOTH cases — the real numbers are the deliverable, not a green checkmark.

## Verify / commit
Run `LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH" python -m pytest tests -v` (all prior 7 + the 2 new pass, or report real numbers). Commit (stage the modified/new files; end message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`). Report to `/home/jianghan/Workspace/diffmpc-learning/.superpowers/sdd/sqp-nlpkkt-report.md` with the conv histories for both cases.
