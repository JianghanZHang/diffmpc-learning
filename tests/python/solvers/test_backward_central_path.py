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
    central_path_nlp_solve,
    central_path_nlp_grad,
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
_POLE_UP = jnp.array([0.0, 0.0, 0.0, 0.0])
_GAMMA = 1.0e2
_HARD_GAMMA = 1.0e8
_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}
_CP = dict(rho_bar=0.1, max_iter=50000, tol=1.0e-11)
WEIGHT_KEYS = [
    "weights_penalization_reference_state_trajectory",  # Q diag (4)
    "weights_penalization_control_squared",             # R diag (1)
]
# Fixed NLP loss: track pole-up; cotangents are explicit.
def _loss(states, controls):
    return 0.5 * jnp.sum((states - _POLE_UP) ** 2) + 0.5e-2 * jnp.sum(controls ** 2)
def _loss_grad(states, controls):
    return jax.grad(_loss, argnums=(0, 1))(states, controls)


def _build_nlp_solver(umax, horizon=12):
    # horizon=12: full backward machinery, but each forward is ~2-3s warm (jit_inner),
    # making convergence-checked FD over all weights tractable. AD==TurboMPC is
    # horizon-independent (verified). umax=1e7 -> interior; umax=2 -> bounds active.
    dynamics, pp = build_cartpole_problem(horizon=horizon, umax=umax, dt=0.04)
    pp["initial_state"] = _POLE_DOWN
    ocp = OptimalControlProblem(dynamics=dynamics, params=pp)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 60
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,   # exact dense KKT ground truth
    )
    return solver, pp


def _flat(d, keys):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in keys])


def _cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _fd_grad_weights(loss_of_w, weights, keys, eps_seq=(1e-2, 1e-3, 1e-4),
                     plateau_rtol=2e-2, atol=1e-7):
    """Convergence-checked central-diff gradient of a scalar loss over a weights dict.

    Returns (flat_grad, flagged_any). Per scalar entry: decreasing eps; the largest eps
    agreeing with the next-smaller (rel OR abs) is the plateau value; entries with no
    plateau are flagged (would indicate a discontinuity / noise floor).
    """
    base = {k: np.array(weights[k], dtype=float) for k in keys}
    grad, flagged = [], False
    for k in keys:
        arr = base[k]
        gk = np.zeros(arr.size)
        for i in range(arr.size):
            vals = []
            for eps in eps_seq:
                wp = {kk: jnp.asarray(base[kk]) for kk in keys}
                wm = {kk: jnp.asarray(base[kk]) for kk in keys}
                ap = arr.copy().reshape(-1); ap[i] += eps
                am = arr.copy().reshape(-1); am[i] -= eps
                wp[k] = jnp.asarray(ap.reshape(arr.shape))
                wm[k] = jnp.asarray(am.reshape(arr.shape))
                vals.append((loss_of_w(wp) - loss_of_w(wm)) / (2 * eps))
            chosen, ok = vals[-1], False
            for a, b in zip(vals[:-1], vals[1:]):
                if abs(a - b) <= plateau_rtol * abs(b) + atol:
                    chosen, ok = a, True
                    break
            flagged = flagged or (not ok)
            gk[i] = chosen
        grad.append(gk)
    return np.concatenate(grad), flagged


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
    d_idx = [(8, NX, NX), (20, 2, 2), (12, 0, 0), (3, 1, 1)]  # nontrivial diagonal entries
    n_nontrivial = 0
    for (t, i, j) in d_idx:
        fd, ok = _converged_fd(loss_D, D0, (t, i, j))
        assert ok, f"FD did not plateau for D[{t},{i},{j}]"
        ad = float(dL_dD[t, i, j])
        assert abs(ad - fd) <= 1e-3 * abs(fd) + 2e-6, f"dL/dD[{t},{i},{j}]: AD={ad:.6e} FD={fd:.6e}"
        n_nontrivial += abs(fd) > 1e-4
    assert n_nontrivial >= 2, "expected several nontrivial dL/dD entries"

    # E (off-diagonal cost coupling): E==0 in the cartpole cost but dL/dE is nontrivial
    # (x* != 0). Perturb an E entry, re-solve. Validates the E branch of the cost VJP.
    @jax.jit
    def loss_E(E):
        qp = dataclasses.replace(qp1, cost=QPCostBlocks(qp1.cost.D, E, qp1.cost.q))
        x, _, _ = solve_qp_central_path(qp, schur, target_kappa=kappa, slack_weight=_GAMMA, **_CP)
        return jnp.sum(x_bar * x)

    E0 = qp1.cost.E
    e_idx = [(5, 0, 0), (10, NX, 1), (15, 2, NX)]
    n_e_nontrivial = 0
    for (t, i, j) in e_idx:
        fd, ok = _converged_fd(loss_E, E0, (t, i, j))
        assert ok, f"FD did not plateau for E[{t},{i},{j}]"
        ad = float(dL_dE[t, i, j])
        assert abs(ad - fd) <= 1e-3 * abs(fd) + 2e-6, f"dL/dE[{t},{i},{j}]: AD={ad:.6e} FD={fd:.6e}"
        n_e_nontrivial += abs(fd) > 1e-4
    assert n_e_nontrivial >= 1, "expected a nontrivial dL/dE entry"


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


# --------------------------------------------------------------------------- #
# Task 2: NLP-level backward, UNBOUNDED (interior) -> AD == TurboMPC == FD
# --------------------------------------------------------------------------- #
_NLP_HARD = dict(slack_weight=_HARD_GAMMA, target_kappa=1e-7, conv_slack_weight=None,
                 kappa_anneal=True, kappa_anneal_start=1e-3, kappa_anneal_factor=0.1,
                 linesearch=False, max_sqp_iter=60, tol=1e-5, jit_inner=True)


def test_task2_nlp_backward_unbounded_matches_turbompc_and_fd():
    # umax=50: control bounds PRESENT (one-sided rows m>0) but INACTIVE (swing-up needs
    # |u|~17 < 50), so this genuinely exercises the inequality path in the interior limit
    # W->0 (vs umax=1e7, which drops the rows entirely, m=0). AD must equal the unrelaxed
    # TurboMPC backward (no active rows) and FD.
    solver, pp = _build_nlp_solver(umax=50.0)
    weights = {k: pp[k] for k in WEIGHT_KEYS}

    # AD: our relaxed NLP backward.
    res, dL_ad, info = central_path_nlp_grad(
        solver, pp, weights, _loss_grad, **_NLP_HARD)
    assert float(res["final_stationarity"]) < 1e-4
    assert float(res["final_eq"]) < 1e-4
    assert float(res["final_ineq"]) < 1e-4
    # Confirm the inequality path is exercised (m>0) AND interior (bounds inactive => W~0).
    qp1 = to_one_sided(solver._build_qp_data(res["states"], res["controls"], pp), _HARD_GAMMA)
    assert qp1.ineq.G.shape[1] > 0, "expected nonempty inequality rows (umax finite)"
    assert float(jnp.max(jnp.abs(res["controls"]))) < 50.0       # interior: no active bound
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, solver.program.horizon, NX, NU, pcg_params=_PCG)
    _xs, _du, _ = solve_qp_central_path(qp1, schur, target_kappa=float(res["kappas"][-1]),
                                        slack_weight=_HARD_GAMMA, **_CP)
    W = relaxed_complementarity_weight(qp1, _xs, _du[2], _HARD_GAMMA)
    assert float(jnp.max(W)) < 1e-6, "interior limit: W should be ~0 on inactive rows"
    g_ad = _flat(dL_ad, WEIGHT_KEYS)

    # Ground truth A: unrelaxed TurboMPC backward (exact dense KKT).
    ig = solver.initial_guess(pp)
    def tm_loss(w):
        sol = solver.solve(ig, pp, w)
        return _loss(sol.states, sol.controls)
    dL_tm = jax.grad(tm_loss)(weights)
    g_tm = _flat(dL_tm, WEIGHT_KEYS)

    # Ground truth B: convergence-checked FD of OUR forward solver.
    def fwd_loss(w):
        r = central_path_nlp_solve(solver, pp, w, **_NLP_HARD)
        return float(_loss(r["states"], r["controls"]))
    g_fd, flagged = _fd_grad_weights(fwd_loss, weights, WEIGHT_KEYS)

    assert not flagged, "FD did not plateau (unexpected discontinuity in interior NLP)"
    assert _cosine(g_ad, g_tm) > 1 - 1e-5, f"AD vs TurboMPC cos={_cosine(g_ad, g_tm)}"
    assert _rel_l2(g_ad, g_tm) < 1e-3, f"AD vs TurboMPC rel_l2={_rel_l2(g_ad, g_tm)}"
    assert _cosine(g_ad, g_fd) > 1 - 1e-4, f"AD vs FD cos={_cosine(g_ad, g_fd)}"
    assert _rel_l2(g_ad, g_fd) < 1e-2, f"AD vs FD rel_l2={_rel_l2(g_ad, g_fd)}"


# --------------------------------------------------------------------------- #
# Task 3: NLP-level backward, BOUNDED (active) -> AD == FD of the SAME relaxed
# solver, at BOTH kappa regimes. NOT compared to TurboMPC: the relaxed and hard
# backwards legitimately differ by O(kappa) at active rows.
# --------------------------------------------------------------------------- #
def _bounded_cfg(target_kappa, kappa_anneal):
    return dict(slack_weight=_GAMMA, target_kappa=target_kappa, conv_slack_weight=_GAMMA,
                kappa_anneal=kappa_anneal, kappa_anneal_start=1e-3, kappa_anneal_factor=0.1,
                linesearch=False, max_sqp_iter=60, tol=1e-5, jit_inner=True)


def _nlp_ad_vs_fd_bounded(cfg):
    solver, pp = _build_nlp_solver(umax=2.0)              # active control bounds
    weights = {k: pp[k] for k in WEIGHT_KEYS}

    res, dL_ad, info = central_path_nlp_grad(solver, pp, weights, _loss_grad, **cfg)
    # Converged relaxed NLP-KKT: dynamics feasible + inner relaxed QP solved + SQP fixed point.
    assert float(res["final_eq"]) < 1e-6
    assert float(info["prim_res"]) < 1e-6
    # Control bounds active -> W is genuinely exercised (not the interior limit).
    assert float(jnp.max(jnp.abs(res["controls"]))) >= 0.99 * 2.0

    g_ad = _flat(dL_ad, WEIGHT_KEYS)

    def fwd_loss(w):
        r = central_path_nlp_solve(solver, pp, w, **cfg)
        return float(_loss(r["states"], r["controls"]))
    g_fd, flagged = _fd_grad_weights(fwd_loss, weights, WEIGHT_KEYS)

    assert not flagged, "FD did not plateau (relaxed map should be C1 at fixed kappa)"
    assert _cosine(g_ad, g_fd) > 1 - 1e-3, f"AD vs FD cos={_cosine(g_ad, g_fd)}"
    assert _rel_l2(g_ad, g_fd) < 1e-2, f"AD vs FD rel_l2={_rel_l2(g_ad, g_fd)}"


def test_task3_nlp_backward_bounded_annealed_kappa():
    # Annealed to small kappa (~1e-6): closest to the true solution; stiff W.
    _nlp_ad_vs_fd_bounded(_bounded_cfg(target_kappa=1e-6, kappa_anneal=True))


def test_task3_nlp_backward_bounded_fixed_kappa():
    # Fixed moderate kappa (1e-4): smoother map, wider FD plateau.
    _nlp_ad_vs_fd_bounded(_bounded_cfg(target_kappa=1e-4, kappa_anneal=False))
