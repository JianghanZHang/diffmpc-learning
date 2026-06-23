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

from .central_path_admm import solve_qp_central_path  # noqa: E402


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
    folded into ``D_aug``. ``x_bar`` is the primal cotangent ``(N+1, n)``. Returns the
    primal adjoint ``lam_x`` ``(N+1, n)``.
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
    (lam_states, lam_controls), _ = solve_backward_kkt(bwd_qp, zshape)
    return pack_x(lam_states, lam_controls)


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
    lam_x = solve_reduced_relaxed_kkt(D_aug, qp1.cost.E, qp1.eq, x_bar)

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
