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
