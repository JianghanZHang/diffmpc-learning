"""SQP outer loop driven by the central-path ADMM as the inner QP solver.

Reuses TurboMPC's SQP machinery on the nonlinear OCP:
- `solver._build_qp_data` linearizes the OCP into a (two-sided box) QP at the iterate;
- the inner QP is solved by `solve_qp_central_path` on the one-sided slack reformulation
  (`to_one_sided`) with a cuDSS Schur backend;
- `backtracking_linesearch` globalizes the SQP step;
- `solver._compute_first_order_convergence_error` is the NLP-KKT residual we drive to tol.

The deliverable is the per-iteration NLP-KKT `conv_history`: showing the central-path inner
solver lets SQP converge the nonlinear problem for both the hard-box and soft-box NLP.
"""
from __future__ import annotations

import dataclasses
import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from turbompc.solvers.admm.admm import ADMMState, _inf_norm, _apply_P, _apply_Ct, _apply_Gt
from turbompc.solvers.linesearch import backtracking_linesearch
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_data import QPInequalityBlocks
from turbompc.solvers.qp_utils import pack_x

from .central_path_admm import to_one_sided, solve_qp_central_path

_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}  # required by make_schur_solver (ignored by cuDSS)


def _conv_check_qp(qp, conv_slack_weight):
    """Original two-sided QP with the slack semantics used by the NLP-KKT check.

    `conv_slack_weight=None` -> hard box (use_slack_variables=False); otherwise the soft
    box with `slack_penalization_weight=conv_slack_weight`. Decoupled from the OCP's slack
    flag so the line search can use the plain (non-slack) cost path on a vanilla
    OptimalControlProblem while the convergence check still sees the soft QP.
    """
    if conv_slack_weight is None:
        ineq = dataclasses.replace(
            qp.ineq, use_slack_variables=False,
            slack_penalization_weight=jnp.asarray(0.0, qp.ineq.u.dtype),
        )
    else:
        ineq = dataclasses.replace(
            qp.ineq, use_slack_variables=True,
            slack_penalization_weight=jnp.asarray(conv_slack_weight, qp.ineq.u.dtype),
        )
    return dataclasses.replace(qp, ineq=ineq)


def sqp_central_path(
    solver,
    problem_params,
    *,
    slack_weight,
    target_kappa,
    max_sqp_iter=60,
    tol=1e-5,
    linesearch=True,
    weights=None,
    rho_bar=0.1,
    cp_max_iter=50000,
    cp_tol=1e-11,
    conv_slack_weight=None,
    kappa_anneal=False,
    kappa_anneal_start=1e-3,
    kappa_anneal_factor=0.1,
    jit_inner=False,
    verbose=False,
):
    """SQP outer loop with the central-path ADMM inner QP solver on the nonlinear OCP.

    Args:
        solver: a constructed `TurboMPCSolver` (provides `_build_qp_data`,
            `_compute_first_order_convergence_error`, `program`, `params`).
        problem_params: OCP problem params dict.
        slack_weight: elastic slack penalty gamma for the inner one-sided central-path solve.
            Hard box: use a large value (e.g. 1e8). Soft box: the soft gamma (e.g. 1e2).
        target_kappa: fixed log-barrier temperature for the inner solve. A fixed kappa>0
            floors the NLP-KKT residual at ~O(kappa); use kappa_anneal to drive it lower.
        conv_slack_weight: slack penalty used by the NLP-KKT check. None => hard box
            (no slack); else the soft gamma (must match the soft `ref`).
        kappa_anneal: if True, start the inner kappa at `kappa_anneal_start` and multiply
            by `kappa_anneal_factor` each SQP iter, clamped at `target_kappa`.

    Returns:
        dict with `states`, `controls`, `conv_history` (jnp.ndarray of NLP-KKT residuals,
        one per SQP iteration), `num_iter`, `final_conv`, `alphas`, `kappas`.
    """
    program = solver.program
    nx = program.num_state_variables
    nu = program.num_control_variables
    N = program.horizon

    states, controls = program.initial_guess(problem_params)
    m_orig = solver._build_qp_data(states, controls, problem_params).ineq.G.shape[1]
    slacks = jnp.zeros((N + 1, m_orig), dtype=states.dtype)

    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)

    # The inner solve's lax.while_loop is re-traced on every eager call; jitting it
    # (compiled once, reused across all SQP iters / kappa values) is numerically
    # identical but ~10x faster, which makes per-iteration backward FD tractable.
    if jit_inner:
        @functools.partial(jax.jit, static_argnums=(1,))
        def _inner(qp_data, schur_solver, kappa):
            return solve_qp_central_path(
                qp_data, schur_solver, target_kappa=kappa, slack_weight=slack_weight,
                rho_bar=rho_bar, max_iter=cp_max_iter, tol=cp_tol)
    else:
        def _inner(qp_data, schur_solver, kappa):
            return solve_qp_central_path(
                qp_data, schur_solver, target_kappa=kappa, slack_weight=slack_weight,
                rho_bar=rho_bar, max_iter=cp_max_iter, tol=cp_tol)

    conv_history = []
    alphas = []
    kappas = []
    kappa = float(kappa_anneal_start) if kappa_anneal else float(target_kappa)

    final_conv = jnp.asarray(jnp.inf, states.dtype)
    final_stationarity = jnp.asarray(jnp.inf, states.dtype)
    final_eq = jnp.asarray(jnp.inf, states.dtype)
    final_ineq = jnp.asarray(jnp.inf, states.dtype)
    it_done = 0
    for it in range(max_sqp_iter):
        if kappa_anneal:
            kappa = max(float(target_kappa), kappa)
        kappas.append(kappa)

        # 1) linearize the OCP into the original two-sided QP at the current iterate.
        qp = solver._build_qp_data(states, controls, problem_params)

        # 2) one-sided slack reformulation; 3) central-path ADMM inner solve.
        qp1 = to_one_sided(qp, slack_weight)
        x_new, (y_f_0, y_f_dyn, y_g_stacked), cp_info = _inner(
            qp1, schur, jnp.asarray(kappa, states.dtype))

        # 4) unpack candidate primal.
        states_new = x_new[:, :nx]
        controls_new = x_new[:, nx:]

        # 5) map stacked one-sided dual -> original two-sided multiplier.
        #    [G;-G]^T [nu_u; nu_l] = G^T (nu_u - nu_l).
        m = qp.ineq.G.shape[1]
        y_g = y_g_stacked[:, :m] - y_g_stacked[:, m:]

        # 6) slacks for the candidate iterate.
        if conv_slack_weight is None:
            slacks_new = jnp.zeros((N + 1, m), dtype=states.dtype)
        else:
            # Soft box: slack stationarity is gamma*s + y_g = 0 -> s = -y_g/gamma.
            # (This is the two-sided soft slack; it equals proj_[l,u](Gx)-Gx at the soln.)
            slacks_new = -y_g / jnp.asarray(conv_slack_weight, states.dtype)

        # 7) globalize.
        if linesearch:
            (states, controls, slacks), alpha = backtracking_linesearch(
                program, problem_params, solver.params,
                states, controls, slacks, states_new, controls_new, slacks_new,
            )
        else:
            states, controls, slacks = states_new, controls_new, slacks_new
            alpha = jnp.asarray(1.0, states.dtype)
        alphas.append(float(alpha))

        # 8) NLP-KKT residual at the accepted iterate.
        qp_at_new = solver._build_qp_data(states, controls, problem_params)
        qp_conv = _conv_check_qp(qp_at_new, conv_slack_weight)
        admm_state = ADMMState(
            x_blocks=pack_x(states, controls),
            y_g=y_g, y_f_0=y_f_0, y_f_dyn=y_f_dyn,
            z_g=jnp.zeros((N + 1, m), dtype=states.dtype),
            xi_g=slacks, rho_bar=jnp.asarray(rho_bar, states.dtype),
        )
        conv, eq_err, ineq_err, *_ = solver._compute_first_order_convergence_error(
            qp_conv, states, controls, slacks, admm_state, problem_params,
        )
        # Compute stationarity separately (not returned by _compute_first_order_convergence_error).
        states_qp, controls_qp = solver.program.scale_states_controls(states, controls, problem_params)
        x_blocks_qp = pack_x(states_qp, controls_qp)
        stat_err = _inf_norm(
            _apply_P(qp_conv, x_blocks_qp)
            + qp_conv.cost.q
            + _apply_Ct(qp_conv, admm_state.y_f_0, admm_state.y_f_dyn)
            + _apply_Gt(qp_conv, admm_state.y_g)
        )
        conv_history.append(float(conv))
        final_conv = conv
        final_eq = eq_err
        final_ineq = ineq_err
        final_stationarity = stat_err
        it_done = it + 1
        if verbose:
            print(f"[sqp it {it:2d}] kappa={kappa:.1e} alpha={float(alpha):.3f} "
                  f"conv={float(conv):.3e} cp_iters={int(cp_info['iters'])} "
                  f"prim={float(cp_info['prim_res']):.2e}")

        if float(conv) < tol:
            break
        if kappa_anneal:
            kappa = kappa * float(kappa_anneal_factor)

    return {
        "states": states,
        "controls": controls,
        "conv_history": jnp.asarray(conv_history),
        "num_iter": it_done,
        "final_conv": float(final_conv),
        "final_stationarity": float(final_stationarity),
        "final_eq": float(final_eq),
        "final_ineq": float(final_ineq),
        "alphas": jnp.asarray(alphas),
        "kappas": jnp.asarray(kappas),
    }
