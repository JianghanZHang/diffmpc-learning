import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from central_path_admm import retraction_map, elastic_retraction  # noqa: E402
from central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402

from tests.helpers.problem_fixtures import cost_blocks_from_qr  # noqa: E402
from turbompc.solvers.qp_data import qpdata_from_ocp_blocks  # noqa: E402
from turbompc.solvers.qp_utils import ZShape  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend, AdmmBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402


def test_retraction_map_complementarity_and_relu_limit():
    v = jnp.asarray(np.linspace(-5, 5, 21))
    g = 0.3
    assert float(jnp.max(jnp.abs(retraction_map(v, g) * retraction_map(-v, g) - g))) < 1e-10
    assert float(jnp.max(jnp.abs(retraction_map(v, 1e-9) - jnp.maximum(v, 0.0)))) < 1e-4


def test_elastic_retraction_inactive_row_recovers_target_as_kappa_to_zero():
    h = jnp.array([2.0])
    z_tilde = jnp.array([0.5])          # below the bound -> inactive
    z_g, xi = elastic_retraction(z_tilde, h, kappa=1e-8, rho=0.3, slack_weight=1.0)
    assert float(jnp.abs(z_g[0] - 0.5)) < 1e-4 and float(xi[0]) < 1e-4


def test_elastic_retraction_active_row_matches_soft_blend_and_slack_engages():
    h = jnp.array([2.0])
    z_tilde = jnp.array([7.0])          # above the bound -> active/violated
    rho, gamma, kappa = 0.3, 0.1, 1e-6
    z_g, xi = elastic_retraction(z_tilde, h, kappa, rho, gamma)
    soft = h[0] + (z_tilde[0] - h[0]) * rho / (gamma + rho)   # frac-blend / soft solution
    assert float(jnp.abs(z_g[0] - soft)) < 1e-3
    assert float(xi[0]) > 0.0 and float(z_g[0]) > float(h[0])  # slack engages, constraint relaxed


_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}  # required kwarg of make_schur_solver (ignored by cuDSS)


def _toy_qp(bound):
    rng = np.random.default_rng(1)
    N, nx, nu = 3, 1, 1
    As_next = jnp.asarray(0.1 * rng.standard_normal((N, nx, nx)))
    Bs_next = jnp.asarray(0.1 * rng.standard_normal((N, nx, nu)))
    As = jnp.asarray(0.1 * rng.standard_normal((N, nx, nx)))
    Bs = jnp.asarray(0.1 * rng.standard_normal((N, nx, nu)))
    Cs = jnp.asarray(0.5 * rng.standard_normal((N + 1, nx)))
    Qm = jnp.tile(jnp.eye(nx)[None], (N + 1, 1, 1)); Rm = jnp.tile(jnp.eye(nu)[None], (N + 1, 1, 1))
    Rd = jnp.tile(jnp.eye(nu)[None], (N, 1, 1)) * 0.05
    qv = jnp.asarray(0.5 * rng.standard_normal((N + 1, nx))); rv = jnp.asarray(0.5 * rng.standard_normal((N + 1, nu)))
    D, E, q = cost_blocks_from_qr(Qm, Rm, Rd, qv, rv)
    A0 = jnp.concatenate([jnp.eye(nx), jnp.zeros((nx, nu))], axis=1)
    lo = jnp.concatenate([-1e7 * jnp.ones((N + 1, nx)), -bound * jnp.ones((N + 1, nu))], -1)
    hi = jnp.concatenate([1e7 * jnp.ones((N + 1, nx)), bound * jnp.ones((N + 1, nu))], -1)
    ineq_blocks = jnp.tile(jnp.eye(nx + nu)[None], (N + 1, 1, 1))
    qp = qpdata_from_ocp_blocks(D=D, E=E, q=q, A0=A0, c0=Cs[0], As_next=As_next, Bs_next=Bs_next,
                                As=As, Bs=Bs, c_dyn=Cs[1:], ineq_blocks=ineq_blocks, ineq_l=lo, ineq_u=hi)
    return qp, N, nx, nu


def test_central_path_loop_converges_on_cudss():
    qp, N, nx, nu = _toy_qp(bound=0.05)
    qp1 = to_one_sided(qp, slack_weight=1e2)
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    x, info = solve_qp_central_path(qp1, schur, target_kappa=1e-4, slack_weight=1e2,
                                    rho_bar=0.1, max_iter=20000, tol=1e-10)
    assert float(info["prim_res"]) < 1e-6        # dynamics + consensus feasible
    assert int(info["iters"]) < 20000            # converged before the cap
    assert jnp.all(jnp.isfinite(x))
