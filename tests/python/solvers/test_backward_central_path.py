"""Backward pass / VJP tests for the central-path ADMM solver.

Task 1 (this file, first block): QP-level relaxed-KKT VJP vs finite differences on
the bounded cartpole one-sided QP (active control-bound rows exercise the smooth
weight W). Tasks 2-3 (NLP-level) are added below as they are implemented.
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import dataclasses

from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path
from diffmpc_learning.solvers.backward import (
    relaxed_complementarity_weight,
    qp_central_path_cost_vjp,
    make_qp_central_path_diff,
)

from turbompc.solvers.qp_data import QPCostBlocks
from turbompc.solvers.qp_utils import ZShape
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver

from benchmark_cartpole_coupling import build_cartpole_problem
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params

NX, NU = 4, 1
_POLE_DOWN = jnp.array([0.0, 0.0, jnp.pi, 0.0])
_GAMMA = 1.0e2
_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}
_CP = dict(rho_bar=0.1, max_iter=50000, tol=1.0e-11)


def _cartpole_one_sided_qp(umax, slack_weight):
    dynamics, pp = build_cartpole_problem(horizon=25, umax=umax, dt=0.04)
    pp["initial_state"] = _POLE_DOWN
    ocp = OptimalControlProblem(dynamics=dynamics, params=pp)
    sp = load_solver_params("turbompc.yaml")
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    )
    ig = solver.initial_guess(pp)
    qp_two_sided = solver._build_qp_data(ig.states, ig.controls, pp)
    return to_one_sided(qp_two_sided, slack_weight=slack_weight)


def _fd_central(loss_fn, x0, idx, eps):
    """Central difference of scalar loss_fn at flat index idx of array x0."""
    xp = x0.at[idx].add(eps)
    xm = x0.at[idx].add(-eps)
    return float((loss_fn(xp) - loss_fn(xm)) / (2.0 * eps))


def _converged_fd(loss_fn, x0, idx, eps_seq=(1e-2, 1e-3, 1e-4, 1e-5),
                  plateau_rtol=1e-2, atol=1e-6):
    """Convergence-checked central-diff (CLAUDE.md): shrink eps to a plateau.

    Decreasing eps; the largest eps that agrees with the next-smaller one (within
    ``plateau_rtol`` relative OR ``atol`` absolute) is the trustworthy plateau value.
    The absolute floor handles near-zero gradients, whose FD is pure solver noise and
    has no meaningful *relative* plateau (without it, a true-zero gradient would be
    mis-flagged as a discontinuity). A genuine discontinuity (FD ~ 1/eps) grows past
    both tolerances and is still flagged. Returns (value, converged_bool).
    """
    vals = [_fd_central(loss_fn, x0, idx, e) for e in eps_seq]
    for a, b in zip(vals[:-1], vals[1:]):
        if abs(a - b) <= plateau_rtol * abs(b) + atol:
            return a, True
    return vals[-1], False


def test_task1_qp_relaxed_kkt_vjp_matches_fd_bounded():
    """QP-level cost VJP (dL/dq, dL/dD) matches convergence-checked FD; W is active."""
    qp1 = _cartpole_one_sided_qp(umax=2.0, slack_weight=_GAMMA)
    N = qp1.cost.D.shape[0] - 1
    n = qp1.cost.D.shape[1]
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)
    kappa = 1e-4

    x_star, duals, info = solve_qp_central_path(
        qp1, schur, target_kappa=kappa, slack_weight=_GAMMA, **_CP)
    assert float(info["prim_res"]) < 1e-6

    # W must be genuinely exercised: control bounds active during swing-up.
    W = relaxed_complementarity_weight(qp1, x_star, duals[2], _GAMMA)
    assert float(jnp.max(W)) > 1.0, "expected some active one-sided rows (W large)"

    # Fixed cotangent -> linear loss L(theta) = <x_bar, x*(theta)>; dL/dx = x_bar.
    rng = np.random.default_rng(0)
    x_bar = jnp.asarray(rng.standard_normal((N + 1, n)))
    dL_dD, dL_dE, dL_dq = qp_central_path_cost_vjp(qp1, x_star, duals, x_bar, slack_weight=_GAMMA)

    # FD reference solver as a function of perturbed q (schur unaffected by q).
    @jax.jit
    def loss_q(q):
        qp = dataclasses.replace(qp1, cost=QPCostBlocks(qp1.cost.D, qp1.cost.E, q))
        x, _, _ = solve_qp_central_path(qp, schur, target_kappa=kappa, slack_weight=_GAMMA, **_CP)
        return jnp.sum(x_bar * x)

    q0 = qp1.cost.q
    # Sample q entries across stages (early/mid/late, state & control components).
    q_idx = [(0, 0), (0, NX), (5, 1), (12, NX), (18, 2), (24, NX)]
    n_nontrivial = 0
    for (t, i) in q_idx:
        fd, ok = _converged_fd(loss_q, q0, (t, i))
        assert ok, f"FD did not plateau for q[{t},{i}] (discontinuity?)"
        ad = float(dL_dq[t, i])
        assert abs(ad - fd) <= 2e-4 * abs(fd) + 2e-6, f"dL/dq[{t},{i}]: AD={ad:.6e} FD={fd:.6e}"
        n_nontrivial += abs(fd) > 1e-4
    assert n_nontrivial >= 2, "expected several nontrivial dL/dq entries"

    # A few D entries (perturb a diagonal entry; schur recomputed inside the solve).
    @jax.jit
    def loss_D(D):
        qp = dataclasses.replace(qp1, cost=QPCostBlocks(D, qp1.cost.E, qp1.cost.q))
        x, _, _ = solve_qp_central_path(qp, schur, target_kappa=kappa, slack_weight=_GAMMA, **_CP)
        return jnp.sum(x_bar * x)

    D0 = qp1.cost.D
    d_idx = [(0, 0, 0), (8, NX, NX), (20, 2, 2)]  # diagonal entries
    for (t, i, j) in d_idx:
        fd, ok = _converged_fd(loss_D, D0, (t, i, j))
        assert ok, f"FD did not plateau for D[{t},{i},{j}]"
        ad = float(dL_dD[t, i, j])
        assert abs(ad - fd) <= 1e-3 * abs(fd) + 2e-6, f"dL/dD[{t},{i},{j}]: AD={ad:.6e} FD={fd:.6e}"


def test_task1_custom_vjp_wrapper_matches_direct_vjp():
    """make_qp_central_path_diff (jax.grad) reproduces the direct VJP gradient."""
    qp1 = _cartpole_one_sided_qp(umax=2.0, slack_weight=_GAMMA)
    N = qp1.cost.D.shape[0] - 1
    n = qp1.cost.D.shape[1]
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)
    kappa = 1e-4

    x_star, duals, _ = solve_qp_central_path(
        qp1, schur, target_kappa=kappa, slack_weight=_GAMMA, **_CP)
    rng = np.random.default_rng(1)
    x_bar = jnp.asarray(rng.standard_normal((N + 1, n)))
    dL_dD_ref, dL_dE_ref, dL_dq_ref = qp_central_path_cost_vjp(
        qp1, x_star, duals, x_bar, slack_weight=_GAMMA)

    solve = make_qp_central_path_diff(
        qp1, schur, slack_weight=_GAMMA, target_kappa=kappa, **_CP)
    loss = lambda D, E, q: jnp.sum(x_bar * solve(D, E, q))
    dL_dD, dL_dE, dL_dq = jax.grad(loss, argnums=(0, 1, 2))(
        qp1.cost.D, qp1.cost.E, qp1.cost.q)

    assert float(jnp.max(jnp.abs(dL_dq - dL_dq_ref))) < 1e-8
    assert float(jnp.max(jnp.abs(dL_dD - dL_dD_ref))) < 1e-8
    assert float(jnp.max(jnp.abs(dL_dE - dL_dE_ref))) < 1e-8
