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

# --- sys.path shim: resolve `turbompc` (and `tests`) from the diffmpc2 checkout ---
_DL_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))  # diffmpc-learning/
_SOLVER_ROOT = os.path.join(_DL_ROOT, "diffmpc2")
if _SOLVER_ROOT not in sys.path:
    sys.path.insert(0, _SOLVER_ROOT)

# Eager cuDSS FFI registration (loads libcudss_blktridi_ffi.so; needs LD_LIBRARY_PATH).
import turbompc.solvers.linear_systems_solvers.cudss_ffi_backend  # noqa: E402,F401

from turbompc.solvers.admm.admm import (  # noqa: E402
    compute_S_Phiinv, compute_gamma, _apply_C_parts, _apply_G,
)
from turbompc.solvers.qp_data import QPData, QPInequalityBlocks  # noqa: E402


def _inf_norm(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(a)) if a.size else jnp.asarray(0.0, a.dtype)


def retraction_map(v, gamma):
    """b_gamma(v) = (v + sqrt(v^2 + 4*gamma))/2.  Stable; b_g(v)*b_g(-v)=gamma; -> max(v,0) as gamma->0."""
    gamma = jnp.asarray(gamma, v.dtype)
    sq = jnp.sqrt(v * v + 4.0 * gamma)
    out = jnp.where(v >= 0.0, 0.5 * (v + sq), 2.0 * gamma / (sq - v))  # stable branch for v<0
    return jnp.where(gamma == 0.0, jnp.maximum(v, 0.0), out)


def elastic_retraction(z_tilde, h, kappa, rho, slack_weight):
    """Closed-form elastic log-barrier retraction for one-sided rows  Gx <= h.

    Slack s = h - Gx + xi >= 0 (barrier -kappa*log s), elastic xi >= 0 penalized by
    gamma = slack_weight. Joint (s, xi) minimizer is closed form:
        r = h - z_tilde;   s = b_Gamma(r), Gamma = kappa*(1/rho + 1/gamma);
        xi = kappa/(gamma*s);   z_g = h - s + xi   (consensus value of Gx).
    Returns (z_g, xi). As kappa->0: inactive rows -> z_g=z_tilde, xi=0; active rows
    -> z_g = h + (z_tilde - h)*rho/(gamma+rho) (the soft/quadratic-penalty solution).
    """
    dtype = z_tilde.dtype
    kappa = jnp.asarray(kappa, dtype)
    rho = jnp.asarray(rho, dtype)
    gamma = jnp.asarray(slack_weight, dtype)
    Gamma = kappa * (1.0 / rho + 1.0 / gamma)
    r = h - z_tilde
    sq = jnp.sqrt(r * r + 4.0 * Gamma)
    s = jnp.where(r >= 0.0, 0.5 * (r + sq), 2.0 * Gamma / (sq - r))        # s = b_Gamma(r) > 0
    s_minus_r = jnp.where(r >= 0.0, 2.0 * Gamma / (sq + r), s - r)         # s - r, stable for r>>0
    xi = kappa / (gamma * s)
    z_g = z_tilde - s_minus_r + xi                                        # == h - s + xi (no cancellation)
    return z_g, xi


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
    max_iter: int = 20000,
    tol: float = 1.0e-9,
):
    """ADMM forward solve with the closed-form elastic retraction z-update (fixed kappa).

    qp_data MUST be one-sided (Gx <= u; use to_one_sided). Identical to TurboMPC's
    ADMM except the inequality z_g-update is `elastic_retraction` instead of a box
    projection. Over-relaxation alpha=1, no adaptive rho. Linear system: cuDSS Schur.
    """
    dtype = qp_data.cost.q.dtype
    Np1, n = qp_data.cost.D.shape[0], qp_data.cost.D.shape[1]
    N = Np1 - 1
    nx = qp_data.eq.A_minus.shape[1]
    n0 = qp_data.eq.A0.shape[0]
    m = qp_data.ineq.G.shape[1]
    rho_bar = jnp.asarray(rho_bar, dtype)
    rho_f = rho_bar * rho_f_factor
    h = qp_data.ineq.u  # one-sided upper bounds (N+1, m)

    schur = compute_S_Phiinv(qp_data, rho_f, sigma, rho_ineq=rho_bar)

    x0 = jnp.zeros((Np1, n), dtype)
    y_g0 = jnp.zeros((Np1, m), dtype)
    y_f_00 = jnp.zeros((n0,), dtype)
    y_f_dyn0 = jnp.zeros((N, nx), dtype)
    if m:
        z_g0, _ = elastic_retraction(_apply_G(qp_data, x0), h, target_kappa, rho_bar, slack_weight)
    else:
        z_g0 = jnp.zeros((Np1, 0), dtype)
    inf = jnp.asarray(jnp.inf, dtype)
    zero = jnp.asarray(0.0, dtype)
    init = (jnp.asarray(0, jnp.int32), x0, y_g0, y_f_00, y_f_dyn0, z_g0, inf, zero)

    def cond(s):
        it, _, _, _, _, _, delta, _ = s
        return jnp.logical_and(it < max_iter, delta > tol)

    def body(s):
        it, x, y_g, y_f_0, y_f_dyn, z_g, _, _ = s
        gammas = compute_gamma(qp_data, x, z_g, y_g, y_f_0, y_f_dyn,
                               rho_f=rho_f, rho_ineq=rho_bar, sigma=sigma)
        x_new, _ = schur_solver.solve(schur, gammas, x)
        Cx0, Cx = _apply_C_parts(qp_data, x_new)
        ineq_vals = _apply_G(qp_data, x_new)
        if m:
            z_tilde = ineq_vals + y_g / rho_bar
            z_g_new, xi_g = elastic_retraction(z_tilde, h, target_kappa, rho_bar, slack_weight)
            y_g_new = y_g + rho_bar * (ineq_vals - z_g_new)
            xi_max = _inf_norm(xi_g)
        else:
            z_g_new, y_g_new, xi_max = z_g, y_g, zero
        y_f_0_new = y_f_0 + rho_f * (Cx0 - qp_data.eq.c0)
        y_f_dyn_new = y_f_dyn + rho_f * (Cx - qp_data.eq.c)
        delta = jnp.maximum(_inf_norm(x_new - x), _inf_norm(z_g_new - z_g))
        return (it + 1, x_new, y_g_new, y_f_0_new, y_f_dyn_new, z_g_new, delta, xi_max)

    it, x, _, _, _, z_g, delta, xi_max = jax.lax.while_loop(cond, body, init)
    Cx0, Cx = _apply_C_parts(qp_data, x)
    prim = jnp.maximum(_inf_norm(Cx0 - qp_data.eq.c0), _inf_norm(Cx - qp_data.eq.c))
    if m:
        prim = jnp.maximum(prim, _inf_norm(_apply_G(qp_data, x) - z_g))
    return x, {"iters": it, "delta": delta, "prim_res": prim, "xi_max": xi_max}
