"""Central-path (closed-form log-barrier retraction + elastic slack) ADMM forward solve.

Reuses TurboMPC's structured Schur x-update (cuDSS); swaps only the inequality
z-update for a closed-form elastic log-barrier retraction over one-sided rows.
Forward solve only.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)  # x64 required (Global Constraints)
import jax.numpy as jnp

# --- sys.path shim: resolve `turbompc` from the CANONICAL external/diffmpc2 checkout ---
# (branch LogBarrier-ADMM-QP: superset of external/turbompc — same sign-corrected solver +
# inequality Hessian, plus the logbarrier ADMM QP backend and cuDSS version guards. NOT the
# vendored diffmpc2/ at repo root — that release-cleanup checkout has the hard-box
# multiplier-sign bug and lacks get_inequality_lagrangian_hessian.)
# 3 dirs up = repo root -> /external/diffmpc2.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SOLVER_ROOT = os.path.join(_REPO_ROOT, "external", "diffmpc2")
if _SOLVER_ROOT not in sys.path:
    sys.path.insert(0, _SOLVER_ROOT)

# Eager cuDSS FFI registration (loads libcudss_blktridi_ffi.so; needs LD_LIBRARY_PATH).
import turbompc.solvers.linear_systems_solvers.cudss_ffi_backend  # noqa: E402,F401

from turbompc.solvers.admm.admm import (  # noqa: E402
    compute_S_Phiinv, compute_gamma, _apply_C_parts, _apply_G,
    ADMMState, ADMMResiduals, _compute_residuals, _residuals_too_large, _update_rho,
)
from turbompc.solvers.qp_data import QPData, QPInequalityBlocks  # noqa: E402

from .retraction import retraction_map, elastic_retraction, _inf_norm  # noqa: F401


def to_one_sided(qp_data: QPData, slack_weight: float, big: float = 1.0e7) -> QPData:
    """Stack the two-sided box l<=Gx<=u into one-sided rows [G;-G]x <= [u;-l].

    Upper bounds become u' = [u; -l]; lower bounds are inert (-big). Enables the
    elastic slack on the QP (use_slack_variables=True, slack_penalization_weight).
    """
    G, l, u = qp_data.ineq.G, qp_data.ineq.l, qp_data.ineq.u   # (N+1,m,n), (N+1,m), (N+1,m)
    G1 = jnp.concatenate([G, -G], axis=1)                      # (N+1, 2m, n)
    u1 = jnp.concatenate([u, -l], axis=1)                      # (N+1, 2m) upper bounds
    l1 = jnp.full_like(u1, -float(big))                        # inert lower bounds
    ineq = QPInequalityBlocks(
        G=G1, l=l1, u=u1,
        slack_penalization_weight=jnp.asarray(slack_weight, u1.dtype),
        use_slack_variables=True,
    )
    return dataclasses.replace(qp_data, ineq=ineq)


def solve_qp_central_path(
    qp_data: QPData,
    schur_solver,
    *,
    target_kappa: float,
    slack_weight: float,
    rho_bar: float = 0.1,
    sigma: float = 1.0e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    rho_min: float = 1.0e-6,
    rho_max: float = 1.0e6,
    adapt_rho_every: int = 25,
    check_termination_every: int = 25,
    adaptive_rho_tolerance: float = 5.0,
    max_iter: int = 20000,
    tol: float = 1.0e-9,
):
    """Accelerated ADMM forward solve with the closed-form elastic retraction z-update.

    A faithful port of TurboMPC's `_solve_jax_loop` (over-relaxation alpha=1.6, OSQP-style
    adaptive rho with Schur rebuild, residual-based termination) whose ONLY difference is the
    inequality z-update: a closed-form elastic log-barrier retraction (fixed `target_kappa`)
    instead of TurboMPC's box projection. `qp_data` MUST be one-sided (`to_one_sided`).
    Linear system: cuDSS Schur.

    `tol` is the **residual** tolerance, used as `eps_abs = eps_rel = tol` in TurboMPC's
    convergence test `r > eps_abs + eps_rel*norm_term` (primal `max(‖Cx−c‖,‖Gx−z‖)`, dual
    `‖Px+q+Cᵀy+Gᵀy_g‖`). This replaces the old step-norm `delta` check (which let infeasible
    iterates through). Adaptive rho follows the JAX path: blocked after convergence.

    Returns ``(x_blocks, duals, info)`` with ``duals = (y_f_0, y_f_dyn, y_g)`` and ``info``
    keys ``iters, prim_res, dual_res, final_rho, xi_max``.
    """
    dtype = qp_data.cost.q.dtype
    Np1, n = qp_data.cost.D.shape[0], qp_data.cost.D.shape[1]
    N = Np1 - 1
    nx = qp_data.eq.A_minus.shape[1]
    n0 = qp_data.eq.A0.shape[0]
    m = qp_data.ineq.G.shape[1]
    rho_bar0 = jnp.asarray(rho_bar, dtype)
    alpha = jnp.asarray(alpha, dtype)
    eps_abs = eps_rel = jnp.asarray(tol, dtype)
    h = qp_data.ineq.u  # one-sided upper bounds (N+1, m)

    schur0 = compute_S_Phiinv(qp_data, rho_bar0 * rho_f_factor, sigma, rho_ineq=rho_bar0)

    state0 = ADMMState(
        x_blocks=jnp.zeros((Np1, n), dtype),
        y_g=jnp.zeros((Np1, m), dtype),
        y_f_0=jnp.zeros((n0,), dtype),
        y_f_dyn=jnp.zeros((N, nx), dtype),
        z_g=jnp.zeros((Np1, m), dtype),
        xi_g=jnp.zeros((Np1, m), dtype),
        rho_bar=rho_bar0,
    )
    inf = jnp.asarray(jnp.inf, dtype)
    one = jnp.asarray(1.0, dtype)
    res0 = ADMMResiduals(inf, inf, inf, inf, one, one)
    carry0 = (jnp.asarray(0, jnp.int32), state0, schur0, res0)

    def cond(carry):
        it, _, _, residuals = carry
        should_check = (it % check_termination_every) == 0
        too_large = _residuals_too_large(residuals, eps_abs, eps_rel)
        keep_going = jnp.logical_or(jnp.logical_not(should_check), too_large)
        cont = jnp.logical_and(it < max_iter, keep_going)
        return jnp.logical_or(cont, it < 1)

    def body(carry):
        it, state, schur, residuals_prev = carry
        rho_f = state.rho_bar * rho_f_factor

        # x-update (cuDSS Schur), residual quantities from the SOLVED x (pre over-relax)
        gammas = compute_gamma(qp_data, state.x_blocks, state.z_g, state.y_g,
                               state.y_f_0, state.y_f_dyn,
                               rho_f=rho_f, rho_ineq=state.rho_bar, sigma=sigma)
        x_solved, _ = schur_solver.solve(schur, gammas, state.x_blocks)
        Cx0, Cx = _apply_C_parts(qp_data, x_solved)
        ineq_vals = _apply_G(qp_data, x_solved)

        # over-relaxation of the carried primal
        x_blocks = alpha * x_solved + (1.0 - alpha) * state.x_blocks

        # z-update: the ONE difference vs TurboMPC -> elastic log-barrier retraction
        if m:
            z_tilde = alpha * ineq_vals + (1.0 - alpha) * state.z_g + state.y_g / state.rho_bar
            z_g, xi = elastic_retraction(z_tilde, h, target_kappa, state.rho_bar, slack_weight)
            xi_g = -xi   # sign: y_g=gamma*xi at the fixed point, but the dual residual uses gamma*xi_g + y_g
        else:
            z_g, xi_g = state.z_g, state.xi_g

        # dual update (over-relaxed)
        y_f_0 = state.y_f_0 + rho_f * alpha * (Cx0 - qp_data.eq.c0)
        y_f_dyn = state.y_f_dyn + rho_f * alpha * (Cx - qp_data.eq.c)
        if m:
            y_g = state.y_g + state.rho_bar * (alpha * ineq_vals + (1.0 - alpha) * state.z_g - z_g)
        else:
            y_g = state.y_g

        new_state = ADMMState(x_blocks=x_blocks, y_g=y_g, y_f_0=y_f_0, y_f_dyn=y_f_dyn,
                              z_g=z_g, xi_g=xi_g, rho_bar=state.rho_bar)

        # residuals + adaptive rho + Schur rebuild (every check_termination_every)
        should_check = (it % check_termination_every) == 0
        residuals = jax.lax.cond(
            should_check, lambda _: _compute_residuals(qp_data, new_state),
            lambda _: residuals_prev, operand=None)

        def _update_rho_and_schur(_):
            rho_cand = _update_rho(state.rho_bar, residuals, rho_min, rho_max)
            ratio = jnp.maximum(rho_cand / state.rho_bar, state.rho_bar / rho_cand)
            converged = jnp.logical_not(_residuals_too_large(residuals, eps_abs, eps_rel))
            keep = jnp.logical_or(it < 2, it % adapt_rho_every != 0)
            keep = jnp.logical_or(keep, converged)
            keep = jnp.logical_or(keep, ratio < adaptive_rho_tolerance)
            rho_new = jnp.where(keep, state.rho_bar, rho_cand)
            schur_new = jax.lax.cond(
                keep, lambda s: s,
                lambda _: compute_S_Phiinv(qp_data, rho_new * rho_f_factor, sigma, rho_ineq=rho_new),
                schur)
            return rho_new, schur_new

        rho_new, schur = jax.lax.cond(
            should_check, _update_rho_and_schur, lambda _: (state.rho_bar, schur), operand=None)

        next_state = new_state._replace(rho_bar=rho_new)
        return (it + 1, next_state, schur, residuals)

    it, state, schur, _ = jax.lax.while_loop(cond, body, carry0)

    final_res = _compute_residuals(qp_data, state)
    duals = (state.y_f_0, state.y_f_dyn, state.y_g)
    info = {
        "iters": it,
        "prim_res": final_res.primal_residual,
        "dual_res": final_res.dual_residual,
        "final_rho": state.rho_bar,
        "xi_max": _inf_norm(state.xi_g),
    }
    return state.x_blocks, duals, info


def _kappa_schedule(kappa_0, kappa_final, beta):
    """Geometric continuation schedule kappa_{j+1}=max(beta*kappa_j, kappa_final) (PrismQP eq 20).
    Returns a static Python list of kappa levels ending at kappa_final."""
    ks, k = [float(kappa_0)], float(kappa_0)
    while k > kappa_final * (1.0 + 1e-12):
        k = max(beta * k, kappa_final)
        ks.append(k)
    return ks


def solve_qp_central_path_continuation(
    qp_data: QPData,
    schur_solver,
    *,
    slack_weight: float,
    kappa_0: float = 1.0e-1,
    kappa_final: float = 1.0e-6,
    beta: float = 0.2,
    iters_per_level: int = 25,
    rho_bar: float = 0.1,
    sigma: float = 1.0e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    rho_min: float = 1.0e-6,
    rho_max: float = 1.0e6,
    adapt_rho_every: int = 10,
    adaptive_rho_tolerance: float = 5.0,
):
    """Continuation-in-κ forward solve (PrismQP §5): geometrically anneal κ from `kappa_0` to
    `kappa_final` (`κ_{j+1}=max(β·κ_j, κ_final)`), taking `iters_per_level` over-relaxed ADMM steps
    at each level, warm-started into the next. Uses the SAME accelerated ADMM step as the fixed-κ
    solver (over-relaxation α, OSQP adaptive ρ + Schur rebuild) at each κ level — fixed ρ alone does
    not converge the inner QP in 25 steps/level. The κ-schedule is the conditioning device on top.

    `qp_data` MUST be one-sided (`to_one_sided`). Forward solve only — for differentiable use,
    PrismQP recommends the fixed-κ mode (`solve_qp_central_path`) and the κ_final backward.
    Returns ``(x_blocks, duals, info)``; ``info`` adds ``kappas`` (the schedule) and ``levels``.
    """
    dtype = qp_data.cost.q.dtype
    Np1, n = qp_data.cost.D.shape[0], qp_data.cost.D.shape[1]
    N = Np1 - 1
    nx = qp_data.eq.A_minus.shape[1]
    n0 = qp_data.eq.A0.shape[0]
    m = qp_data.ineq.G.shape[1]
    rho_bar0 = jnp.asarray(rho_bar, dtype)
    alpha = jnp.asarray(alpha, dtype)
    eps_abs = eps_rel = jnp.asarray(kappa_final * 1e-3, dtype)   # keep adapting ρ until well below κ_final
    h = qp_data.ineq.u
    kappas = jnp.asarray(_kappa_schedule(kappa_0, kappa_final, beta), dtype)   # (L,) static length

    schur0 = compute_S_Phiinv(qp_data, rho_bar0 * rho_f_factor, sigma, rho_ineq=rho_bar0)
    state0 = ADMMState(
        x_blocks=jnp.zeros((Np1, n), dtype), y_g=jnp.zeros((Np1, m), dtype),
        y_f_0=jnp.zeros((n0,), dtype), y_f_dyn=jnp.zeros((N, nx), dtype),
        z_g=jnp.zeros((Np1, m), dtype), xi_g=jnp.zeros((Np1, m), dtype), rho_bar=rho_bar0)

    def admm_step(carry, kappa):                            # accelerated step at fixed κ level
        it, state, schur = carry
        rho_f = state.rho_bar * rho_f_factor
        gammas = compute_gamma(qp_data, state.x_blocks, state.z_g, state.y_g,
                               state.y_f_0, state.y_f_dyn, rho_f=rho_f, rho_ineq=state.rho_bar, sigma=sigma)
        x_solved, _ = schur_solver.solve(schur, gammas, state.x_blocks)
        Cx0, Cx = _apply_C_parts(qp_data, x_solved)
        ineq_vals = _apply_G(qp_data, x_solved)
        x_blocks = alpha * x_solved + (1.0 - alpha) * state.x_blocks
        if m:
            z_tilde = alpha * ineq_vals + (1.0 - alpha) * state.z_g + state.y_g / state.rho_bar
            z_g, xi = elastic_retraction(z_tilde, h, kappa, state.rho_bar, slack_weight)
            xi_g = -xi
            y_g = state.y_g + state.rho_bar * (alpha * ineq_vals + (1.0 - alpha) * state.z_g - z_g)
        else:
            z_g, xi_g, y_g = state.z_g, state.xi_g, state.y_g
        y_f_0 = state.y_f_0 + rho_f * alpha * (Cx0 - qp_data.eq.c0)
        y_f_dyn = state.y_f_dyn + rho_f * alpha * (Cx - qp_data.eq.c)
        ns = ADMMState(x_blocks=x_blocks, y_g=y_g, y_f_0=y_f_0, y_f_dyn=y_f_dyn,
                       z_g=z_g, xi_g=xi_g, rho_bar=state.rho_bar)
        residuals = _compute_residuals(qp_data, ns)

        def _upd(_):
            rho_c = _update_rho(state.rho_bar, residuals, rho_min, rho_max)
            ratio = jnp.maximum(rho_c / state.rho_bar, state.rho_bar / rho_c)
            converged = jnp.logical_not(_residuals_too_large(residuals, eps_abs, eps_rel))
            keep = jnp.logical_or(jnp.logical_or(it < 2, converged), ratio < adaptive_rho_tolerance)
            rho_new = jnp.where(keep, state.rho_bar, rho_c)
            schur_new = jax.lax.cond(keep, lambda s: s,
                lambda _: compute_S_Phiinv(qp_data, rho_new * rho_f_factor, sigma, rho_ineq=rho_new), schur)
            return rho_new, schur_new
        do_adapt = (it % adapt_rho_every) == 0
        rho_new, schur = jax.lax.cond(do_adapt, _upd, lambda _: (state.rho_bar, schur), operand=None)
        return (it + 1, ns._replace(rho_bar=rho_new), schur)

    def level(carry, kappa):                                # iters_per_level steps at fixed κ level
        carry = jax.lax.fori_loop(0, iters_per_level, lambda _i, c: admm_step(c, kappa), carry)
        return carry, None

    (it, state, schur), _ = jax.lax.scan(level, (jnp.asarray(0, jnp.int32), state0, schur0), kappas)

    final_res = _compute_residuals(qp_data, state)
    duals = (state.y_f_0, state.y_f_dyn, state.y_g)
    info = {
        "iters": int(len(_kappa_schedule(kappa_0, kappa_final, beta))) * iters_per_level,
        "levels": len(_kappa_schedule(kappa_0, kappa_final, beta)),
        "kappas": kappas,
        "prim_res": final_res.primal_residual,
        "dual_res": final_res.dual_residual,
        "final_kappa": kappas[-1],
        "xi_max": _inf_norm(state.xi_g),
    }
    return state.x_blocks, duals, info
