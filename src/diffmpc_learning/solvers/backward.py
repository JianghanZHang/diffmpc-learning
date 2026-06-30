"""κ-relaxed (central-path) backward pass / VJP for the central-path ADMM solver.

Differentiates the *same* relaxed fixed point the forward solver converges to, so the
gradient is smooth across inequality active-set changes (unlike the hard active-set
backward). The reduced relaxed-KKT linearization folds the inequalities into the
Hessian via a smooth per-row weight ``W`` (replacing TurboMPC's hard active mask):

    forward fixed point (per one-sided row i of  G1 x <= h,  gamma = slack_weight):
        s_i = h_i - (G1 x)_i + y_g,i/gamma      (barrier slack, s_i*y_g,i = kappa)
        W_i = y_g,i / (s_i + y_g,i/gamma)

    reduced symmetric backward KKT  (lam_x = primal adjoint, lam_f = eq-dual adjoint):
        [[ P + G1^T diag(W) G1,  C^T ],   [lam_x]     [ dL/dx ]
         [ C,                    0   ]] * [lam_f]  =  [   0   ]

    cost-only weight gradient:
        dL/d(theta) = - lam_x . d/d(theta) [ P(D,E) x* + q ]

As kappa->0: inactive row y_g->0 => W->0 (drops out); soft-active => W->gamma;
gamma->inf hard-active => W->inf (enforces G1 dx = 0). The interior (no active row)
case recovers TurboMPC's equality-only backward exactly.

Forward solve only is reused (cuDSS Schur); the backward KKT is a dense solve.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)  # x64 required (Global Constraints)
import jax.numpy as jnp

from turbompc.solvers.admm.admm import _apply_block_tridiag, _apply_G  # noqa: E402
from turbompc.solvers.qp_data import (  # noqa: E402
    QPData, QPCostBlocks, QPEqualityBlocks, QPInequalityBlocks,
)
from turbompc.solvers.qp_utils import ZShape, pack_x  # noqa: E402
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402

from .central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402
from .sqp import sqp_central_path, _PCG  # noqa: E402


# ----------------------------------------------------------------------------- #
# Relaxed-complementarity weight + Hessian augmentation
# ----------------------------------------------------------------------------- #
def relaxed_complementarity_weight(qp1, x_star, y_g_stacked, slack_weight):
    """Smooth per-(one-sided-row) weight ``W_i = y_g,i / (s_i + y_g,i/gamma)``.

    ``qp1`` is the one-sided QP (``to_one_sided`` output); ``x_star`` the converged
    primal; ``y_g_stacked`` the converged one-sided dual ``(N+1, 2m)``. Returns
    ``W`` of shape ``(N+1, 2m)``.
    """
    h = qp1.ineq.u                              # (N+1, 2m) one-sided upper bounds
    Gx = _apply_G(qp1, x_star)                  # (N+1, 2m)
    gamma = jnp.asarray(slack_weight, x_star.dtype)
    s = h - Gx + y_g_stacked / gamma            # barrier slack (> 0)
    return y_g_stacked / (s + y_g_stacked / gamma)


def augment_D_with_relaxed_ineq(D, G1, W):
    """``D_aug[t] = D[t] + G1[t]^T diag(W[t]) G1[t]`` (block-diagonal per stage).

    D (N+1, n, n); G1 (N+1, 2m, n); W (N+1, 2m). Stays block-tridiagonal (only D
    diagonal blocks are touched).
    """
    def per_stage(Dt, Gt, Wt):
        return Dt + Gt.T @ (Wt[:, None] * Gt)
    return jax.vmap(per_stage)(D, G1, W)


# ----------------------------------------------------------------------------- #
# Reduced relaxed-KKT adjoint solve
# ----------------------------------------------------------------------------- #
def solve_reduced_relaxed_kkt(D_aug, E, eq_blocks, x_bar):
    """Solve ``[[P+G1^T W G1, C^T],[C,0]][lam_x; lam_f] = [x_bar; 0]``.

    Reuses TurboMPC's dense ``solve_backward_kkt`` on an empty-inequality homogeneous
    backward QP: cost ``q = -x_bar`` (so the KKT RHS ``-q = x_bar``), the inequality is
    folded into ``D_aug``. ``x_bar`` is the primal cotangent ``(N+1, n)``.

    Returns:
        ``(lam_x, bwd_multipliers)`` where

        * ``lam_x`` ``(N+1, n)`` is the primal adjoint (states + controls stacked).
        * ``bwd_multipliers`` is the flat 1-D backward equality+inequality dual from
          ``solve_backward_kkt``, ordered ``[init-cond n0, dynamics N*nx, inequality]``.
          Because the backward QP is constructed with an empty inequality block,
          the length is ``n0 + N*nx`` (with ``n0 = eq_blocks.A0.shape[0]``, equal to
          ``nx`` when ``constrain_initial_control`` is False). The initial-condition
          slice ``bwd_multipliers[:n0]`` is ``dL/dx0`` — the initial-state cotangent
          needed for closed-loop BPTT through the central-path solver.
    """
    Np1, n = D_aug.shape[0], D_aug.shape[1]
    N = Np1 - 1
    nx = eq_blocks.A_minus.shape[1]
    nu = n - nx
    eq0 = QPEqualityBlocks(
        A0=eq_blocks.A0, A_minus=eq_blocks.A_minus, A_plus=eq_blocks.A_plus,
        c0=jnp.zeros_like(eq_blocks.c0), c=jnp.zeros_like(eq_blocks.c),
    )
    empty_ineq = QPInequalityBlocks(
        G=jnp.zeros((Np1, 0, n), x_bar.dtype),
        l=jnp.zeros((Np1, 0), x_bar.dtype),
        u=jnp.zeros((Np1, 0), x_bar.dtype),
    )
    bwd_qp = QPData(
        cost=QPCostBlocks(D=D_aug, E=E, q=-x_bar), eq=eq0, ineq=empty_ineq,
    )
    zshape = ZShape(horizon=N, num_states=nx, num_controls=nu)
    (lam_states, lam_controls), bwd_mult = solve_backward_kkt(bwd_qp, zshape)
    return pack_x(lam_states, lam_controls), bwd_mult


# ----------------------------------------------------------------------------- #
# QP-level cost VJP
# ----------------------------------------------------------------------------- #
def qp_central_path_cost_vjp(qp1, x_star, duals, x_bar, *, slack_weight):
    """VJP of ``solve_qp_central_path`` w.r.t. the cost blocks ``(D, E, q)``.

    ``qp1`` one-sided QP; ``x_star``/``duals`` the converged forward solution
    (``duals = (y_f_0, y_f_dyn, y_g_stacked)``); ``x_bar`` the cotangent ``dL/dx``
    ``(N+1, n)``. Returns ``(dL_dD, dL_dE, dL_dq)``.
    """
    _, _, y_g_stacked = duals
    W = relaxed_complementarity_weight(qp1, x_star, y_g_stacked, slack_weight)
    D_aug = augment_D_with_relaxed_ineq(qp1.cost.D, qp1.ineq.G, W)
    lam_x, _ = solve_reduced_relaxed_kkt(D_aug, qp1.cost.E, qp1.eq, x_bar)

    # dL/d(D,E,q) = - lam_x . d/d(D,E,q)[ P(D,E) x* + q ]   (only cost depends on theta)
    def cost_stationarity(D, E, q):
        return _apply_block_tridiag(D, E, x_star) + q

    _, vjp_fn = jax.vjp(cost_stationarity, qp1.cost.D, qp1.cost.E, qp1.cost.q)
    dL_dD, dL_dE, dL_dq = vjp_fn(lam_x)
    return (-dL_dD, -dL_dE, -dL_dq)


def make_qp_central_path_diff(qp1, schur_solver, *, slack_weight, **cp_kwargs):
    """Factory: a ``jax.custom_vjp`` of ``(D, E, q) -> x_star`` for ``qp1``.

    Captures the non-differentiable pieces (``qp1`` eq/ineq, ``schur_solver``, solver
    kwargs) by closure and exposes only the differentiable cost blocks, so it composes
    under ``jax.grad``/``jit``. Backward = ``qp_central_path_cost_vjp``.
    """
    import dataclasses

    @jax.custom_vjp
    def solve(D, E, q):
        qp = dataclasses.replace(qp1, cost=QPCostBlocks(D, E, q))
        x, _, _ = solve_qp_central_path(
            qp, schur_solver, slack_weight=slack_weight, **cp_kwargs)
        return x

    def solve_fwd(D, E, q):
        qp = dataclasses.replace(qp1, cost=QPCostBlocks(D, E, q))
        x, duals, _ = solve_qp_central_path(
            qp, schur_solver, slack_weight=slack_weight, **cp_kwargs)
        return x, (qp, x, duals)

    def solve_bwd(res, x_bar):
        qp, x_star, duals = res
        return qp_central_path_cost_vjp(qp, x_star, duals, x_bar, slack_weight=slack_weight)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


# ----------------------------------------------------------------------------- #
# NLP-level backward (exact Lagrangian Hessian + relaxed inequality)
# ----------------------------------------------------------------------------- #
def central_path_nlp_solve(solver, problem_params, weights, *, slack_weight, target_kappa,
                           **sqp_kwargs):
    """Run ``sqp_central_path`` on the weighted problem; return the forward result dict.

    ``weights`` is merged into ``problem_params`` via ``make_params_with_weights`` (the
    same merge TurboMPC uses), so the same call serves both the analytic solve and the
    finite-difference loss.
    """
    pp_w = solver.make_params_with_weights(weights, problem_params)
    return sqp_central_path(solver, pp_w, slack_weight=slack_weight,
                            target_kappa=target_kappa, **sqp_kwargs)


def _relaxed_nlp_backward(solver, problem_params, weights, states_c, controls_c, final_kappa,
                          dL_dstates, dL_dcontrols, *, slack_weight, rho_bar, cp_max_iter,
                          cp_tol, include_ineq_hessian):
    """Relaxed NLP-KKT adjoint at a converged central-path iterate.

    Steps (2)-(4) of the relaxed NLP backward, factored out of ``central_path_nlp_grad``
    so the same machinery serves both the explicit gradient API and the ``custom_vjp``:

      2. re-solve the inner QP once at the converged iterate (``final_kappa``) for a
         consistent ``(x*, duals)``;
      3. backward Hessian = exact Lagrangian Hessian (``D + lambda^T nabla^2 f`` via
         ``get_dynamics_lagrangian_hessian``) ``+ G1^T diag(W) G1`` (relaxed inequality);
      4. solve the reduced KKT for ``(lam_x, bwd_mult)``; weight gradient
         ``dL/dw = - d/dw [ sum grad_cost(x*; w) . lam_x ]`` (cost-only mixed partial),
         and the initial-state cotangent ``dL/dx0 = bwd_mult[:nx]`` (the init-condition
         adjoint dual) for closed-loop BPTT.

    ``(dL_dstates, dL_dcontrols)`` is the primal cotangent ``dL/d(states, controls)``.
    Returns ``(dL_dweights, dL_dx_init, info)`` where ``info`` carries the inner-solve
    diagnostics (with ``dL_dx_init`` added under that key).
    """
    program = solver.program
    nx = program.num_state_variables
    nu = program.num_control_variables
    N = program.horizon

    pp_w = solver.make_params_with_weights(weights, problem_params)

    # (2) consistent primal-dual at the converged linearization.
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    qp = solver._build_qp_data(states_c, controls_c, pp_w)
    qp1 = to_one_sided(qp, slack_weight)
    x_star, duals, info = solve_qp_central_path(
        qp1, schur, target_kappa=final_kappa, slack_weight=slack_weight,
        rho_bar=rho_bar, max_iter=cp_max_iter, tol=cp_tol)
    _, y_f_dyn, y_g_stacked = duals

    # (3) exact Lagrangian Hessian (dynamics + inequality curvature) + relaxed inequality fold.
    dyn_hess = program.get_dynamics_lagrangian_hessian(states_c, controls_c, pp_w, y_f_dyn)
    D_full = qp1.cost.D + dyn_hess
    # Inequality-constraint curvature sum_i mu_i grad^2 g_i (turbompc method, mirrors the
    # dynamics-Hessian call above). mu = net two-sided multiplier nu_u - nu_l (sqp.py:150
    # convention) from the one-sided stacked dual. Zero for linear rows (grad^2 g = 0);
    # nonzero only for nonlinear constraints (e.g. obstacle).
    if include_ineq_hessian:
        m = qp.ineq.G.shape[1]
        y_g_net = y_g_stacked[:, :m] - y_g_stacked[:, m:]
        ineq_hess = program.get_inequality_lagrangian_hessian(states_c, controls_c, pp_w, y_g_net)
        D_full = D_full + ineq_hess
    W = relaxed_complementarity_weight(qp1, x_star, y_g_stacked, slack_weight)
    D_aug = augment_D_with_relaxed_ineq(D_full, qp1.ineq.G, W)

    # (4) adjoint solve + cost-only weight gradient + initial-state cotangent.
    x_bar = pack_x(dL_dstates, dL_dcontrols)
    lam_x, bwd_mult = solve_reduced_relaxed_kkt(D_aug, qp1.cost.E, qp1.eq, x_bar)
    lam_states, lam_controls = lam_x[:, :nx], lam_x[:, nx:]

    def contracted(w):
        params = solver.make_params_with_weights(w, problem_params)
        fx, fu = jax.grad(
            lambda s, c: program.cost(s, c, params), argnums=(0, 1))(states_c, controls_c)
        return jnp.sum(fx * lam_states) + jnp.sum(fu * lam_controls)

    dL_dweights = jax.tree_util.tree_map(lambda g: -g, jax.grad(contracted)(weights))

    # dL/dx0 = init-condition adjoint dual (bwd_mult[:nx]; n0 == nx here). Sign is the
    # SAME positive convention turbompc uses (turbompc_solver.py:766 dL_dx_init = y_f_0_lin[:nx]),
    # verified by finite differences (test_central_path_x0_sensitivity.py). Rescale by
    # 1/state_diff only under variable rescaling (mirrors _build_problem_params_cotangent).
    dL_dx_init = bwd_mult[:nx]
    if program.rescale_optimization_variables:
        _, _, _, _, state_diff, _ = program._get_rescaling_params(pp_w)
        dL_dx_init = dL_dx_init / state_diff

    info = dict(info)
    info["dL_dx_init"] = dL_dx_init
    return dL_dweights, dL_dx_init, info


def central_path_nlp_grad(solver, problem_params, weights, loss_grad_fn, *,
                          slack_weight, target_kappa, rho_bar=0.1,
                          cp_max_iter=50000, cp_tol=1.0e-11, include_ineq_hessian=True,
                          **sqp_kwargs):
    """``dL/dweights`` at the converged NLP solution of the central-path SQP.

    Differentiates the *relaxed* NLP-KKT at the converged primal-dual point:

      1. run ``sqp_central_path`` (weighted) to a converged NLP-KKT;
      2-4. ``_relaxed_nlp_backward``: re-solve the inner QP at the converged iterate,
           build the exact Lagrangian + relaxed-inequality Hessian, solve the reduced
           KKT and contract for the cost-only weight gradient.

    ``loss_grad_fn(states, controls) -> (dL_dstates, dL_dcontrols)``. Returns
    ``(res, dL_dweights, info)`` where ``res`` is the forward dict and ``info`` the
    re-solve diagnostics (now also carrying ``dL_dx_init``; non-breaking).
    """
    pp_w = solver.make_params_with_weights(weights, problem_params)
    res = sqp_central_path(solver, pp_w, slack_weight=slack_weight, target_kappa=target_kappa,
                           rho_bar=rho_bar, cp_max_iter=cp_max_iter, cp_tol=cp_tol, **sqp_kwargs)
    states_c, controls_c = res["states"], res["controls"]
    final_kappa = float(res["kappas"][-1])

    dL_dstates, dL_dcontrols = loss_grad_fn(states_c, controls_c)
    dL_dweights, _dL_dx_init, info = _relaxed_nlp_backward(
        solver, problem_params, weights, states_c, controls_c, final_kappa,
        dL_dstates, dL_dcontrols, slack_weight=slack_weight, rho_bar=rho_bar,
        cp_max_iter=cp_max_iter, cp_tol=cp_tol, include_ineq_hessian=include_ineq_hessian)
    return res, dL_dweights, info


def _build_problem_params_cotangent(solver, problem_params, dL_dx_init):
    """Zeros-everywhere ``problem_params`` cotangent tree with ``initial_state`` set.

    Mirrors turbompc's ``_build_problem_params_cotangent`` (turbompc_solver.py:989): a
    ``tree_map`` over ``problem_params`` that zeros array/float leaves and drops
    int/other leaves to ``None`` (so the pytree structure round-trips through
    ``jax.custom_vjp`` with non-differentiable leaves), then sets ``initial_state`` to
    ``dL_dx_init``. ``dL_dx_init`` is already rescaled in ``_relaxed_nlp_backward``.
    """
    def _zero_cotangent(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jnp.zeros_like(x)
        if isinstance(x, float):
            return jnp.asarray(0.0, dtype=jnp.float64)
        return None

    cotangent = jax.tree_util.tree_map(_zero_cotangent, problem_params)
    cotangent["initial_state"] = dL_dx_init
    return cotangent


def make_central_path_diff(solver, *, slack_weight, target_kappa, include_ineq_hessian=True,
                           rho_bar=0.1, cp_max_iter=50000, cp_tol=1.0e-11, **sqp_kwargs):
    """Factory: a ``jax.custom_vjp`` ``solve(problem_params, weights) -> (states, controls)``.

    Differentiable w.r.t. the cost ``weights`` AND ``problem_params['initial_state']`` (the
    ``dL/dx0`` needed for closed-loop SHAC-style BPTT through ``du*_0/dx0``). The forward is
    the eager ``central_path_nlp_solve`` (an SQP Python loop — it runs concretely inside the
    ``custom_vjp`` ``fwd``); the backward is ``_relaxed_nlp_backward`` (relaxed NLP-KKT adjoint).
    Mirrors turbompc's differentiable-solve pattern (turbompc_solver.py:1019).
    """
    @jax.custom_vjp
    def solve(problem_params, weights):
        res = central_path_nlp_solve(
            solver, problem_params, weights, slack_weight=slack_weight,
            target_kappa=target_kappa, rho_bar=rho_bar, cp_max_iter=cp_max_iter,
            cp_tol=cp_tol, **sqp_kwargs)
        return res["states"], res["controls"]

    def solve_fwd(problem_params, weights):
        res = central_path_nlp_solve(
            solver, problem_params, weights, slack_weight=slack_weight,
            target_kappa=target_kappa, rho_bar=rho_bar, cp_max_iter=cp_max_iter,
            cp_tol=cp_tol, **sqp_kwargs)
        residual = (res["states"], res["controls"], float(res["kappas"][-1]),
                    problem_params, weights)
        return (res["states"], res["controls"]), residual

    def solve_bwd(residual, cot):
        d_states, d_controls = cot
        states_c, controls_c, final_kappa, problem_params, weights = residual
        dL_dweights, dL_dx_init, _info = _relaxed_nlp_backward(
            solver, problem_params, weights, states_c, controls_c, final_kappa,
            d_states, d_controls, slack_weight=slack_weight, rho_bar=rho_bar,
            cp_max_iter=cp_max_iter, cp_tol=cp_tol, include_ineq_hessian=include_ineq_hessian)
        problem_params_cotangent = _build_problem_params_cotangent(
            solver, problem_params, dL_dx_init)
        return (problem_params_cotangent, dL_dweights)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
