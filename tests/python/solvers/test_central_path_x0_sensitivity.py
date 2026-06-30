"""Test that solve_reduced_relaxed_kkt returns (lam_x, bwd_multipliers).

Task A2: expose the backward equality dual from solve_reduced_relaxed_kkt so that
future jax.custom_vjp can compute dL/dx0 (closed-loop BPTT) from the multipliers
corresponding to the initial-condition constraint.

Setup reuses the obstacle solver from test_inequality_hessian.py. We replicate
the block-assembly lines from central_path_nlp_grad (using the initial guess for
states/controls — correctness of the NLP solution is irrelevant here; this is a
shape/interface test).

cuDSS backend required. Run with:
  export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  PYTHONPATH=external/turbompc <venv>/bin/python -m pytest tests/python/solvers/test_central_path_x0_sensitivity.py -v
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path
from diffmpc_learning.solvers.backward import (
    relaxed_complementarity_weight,
    augment_D_with_relaxed_ineq,
    solve_reduced_relaxed_kkt,
    central_path_nlp_solve,
    central_path_nlp_grad,
    make_central_path_diff,
)
from diffmpc_learning.solvers.sqp import _PCG

from turbompc.solvers.qp_utils import pack_x
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver

# ---- replicate _CFG and _build_obstacle_solver from test_inequality_hessian.py ----
from benchmark_problem_setup import build_turbompc_linear_problem
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params

NX, NU = 4, 2
START = jnp.array([0.0, 0.0, 0.0, 0.0])
GOAL = jnp.array([2.0, 1.5, 0.0, 0.0])
OBS_C = jnp.array([1.0, 0.55])
OBS_R = 0.55
HORIZON, DT = 14, 0.18
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"

_CFG = dict(slack_weight=1e4, target_kappa=1e-5, conv_slack_weight=1e4,
            kappa_anneal=True, kappa_anneal_start=1e-2, kappa_anneal_factor=0.3,
            linesearch=False, max_sqp_iter=60, tol=1e-6, jit_inner=True)

# Continuous-time double-integrator matrices used by the obstacle fixture (== the A_c,B_c
# in _build_obstacle_solver). Used for the 2-step BPTT gate's explicit Euler feedback.
A_C = jnp.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
B_C = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
WEIGHT_KEYS = [QK, RK]


def _build_obstacle_solver():
    dyn, base = build_turbompc_linear_problem(horizon=HORIZON, umax=1e4, n_state=NX, n_ctrl=NU)
    A_c = jnp.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    B_c = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    pp = dict(base)
    pp["discretization_resolution"] = DT
    pp["initial_state"] = START
    pp["reference_state_trajectory"] = jnp.tile(GOAL, (HORIZON + 1, 1))
    pp[QK] = jnp.array([5.0, 5.0, 0.0, 0.0])
    pp[RK] = jnp.array([0.1, 0.1])
    pp["weights_penalization_final_state"] = jnp.zeros((NX,))
    pp["control_min_bounds"] = -1e4 * jnp.ones((NU,))
    pp["control_max_bounds"] = 1e4 * jnp.ones((NU,))
    pp["dynamics_state_dot_params"] = {"A": A_c, "B": B_c, "b": jnp.zeros((NX,))}
    pp["obstacles_centers"] = jnp.tile(OBS_C[None, None, :], (HORIZON + 1, 1, 1))
    pp["obstacles_radii"] = jnp.array([OBS_R])
    cons = dict(pp)
    cons["obstacles_dimension"] = 2
    cons["rescale_optimization_variables"] = False
    ocp = OptimalControlProblemObstacle(dynamics=dyn, params=cons)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 60
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    return solver, cons


def test_solve_reduced_relaxed_kkt_returns_two_tuple():
    """solve_reduced_relaxed_kkt must return (lam_x, bwd_multipliers), not a single array.

    Replicates the block-assembly from central_path_nlp_grad using the initial guess
    (no NLP convergence required — this is a shape/interface test). Asserts:
      1. Return value is a 2-tuple.
      2. lam_x has shape (N+1, nx+nu).
      3. bwd_multipliers is 1-D with length == n0 + N*nx
         (init-cond block + dynamics blocks; inequality is empty in solve_reduced_relaxed_kkt).
    """
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in [QK, RK]}
    pp_w = solver.make_params_with_weights(weights, pp)

    N = solver.program.horizon
    nx = solver.program.num_state_variables
    nu = solver.program.num_control_variables
    slack_weight = _CFG["slack_weight"]

    # Use initial guess as states/controls (no NLP convergence needed for shape test).
    ig = solver.initial_guess(pp_w)
    states_c, controls_c = ig.states, ig.controls

    # Replicate block-assembly from central_path_nlp_grad.
    qp = solver._build_qp_data(states_c, controls_c, pp_w)
    qp1 = to_one_sided(qp, slack_weight)

    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    x_star, duals, _ = solve_qp_central_path(
        qp1, schur,
        target_kappa=_CFG["target_kappa"],
        slack_weight=slack_weight,
        rho_bar=0.1, max_iter=5000, tol=1e-6)
    _, _, y_g_stacked = duals

    W = relaxed_complementarity_weight(qp1, x_star, y_g_stacked, slack_weight)
    D_aug = augment_D_with_relaxed_ineq(qp1.cost.D, qp1.ineq.G, W)

    # Arbitrary cotangent x_bar.
    x_bar = pack_x(
        jnp.ones((N + 1, nx), dtype=jnp.float64),
        jnp.ones((N + 1, nu), dtype=jnp.float64),
    )

    # --- THE CALL UNDER TEST ---
    result = solve_reduced_relaxed_kkt(D_aug, qp1.cost.E, qp1.eq, x_bar)

    # Assert: 2-tuple.
    assert isinstance(result, tuple) and len(result) == 2, (
        f"Expected 2-tuple, got {type(result)} of len "
        f"{len(result) if hasattr(result, '__len__') else 'N/A'}"
    )

    lam_x, bwd_mult = result

    # Assert: lam_x shape.
    n = nx + nu
    assert lam_x.shape == (N + 1, n), (
        f"lam_x shape mismatch: expected ({N+1}, {n}), got {lam_x.shape}"
    )

    # Assert: bwd_mult is 1-D with length n0 + N*nx.
    n0 = qp1.eq.A0.shape[0]  # == nx when constrain_initial_control is False
    expected_mult_len = n0 + N * nx
    assert bwd_mult.ndim == 1, f"bwd_mult should be 1-D, got ndim={bwd_mult.ndim}"
    assert bwd_mult.shape[0] == expected_mult_len, (
        f"bwd_mult length mismatch: expected {expected_mult_len} "
        f"(n0={n0} + N*nx={N}*{nx}={N*nx}), got {bwd_mult.shape[0]}"
    )


# --------------------------------------------------------------------------- #
# Task A3: make_central_path_diff custom_vjp differentiable w.r.t. weights AND
# problem_params['initial_state'] (the dL/dx0 needed for closed-loop BPTT).
# Finite differences are the arbiter, especially for the dL/dx0 SIGN (gate 3).
# --------------------------------------------------------------------------- #
def _loss(states, controls):
    return 0.5 * jnp.sum((states[:, :2] - GOAL[:2]) ** 2) + 0.5e-2 * jnp.sum(controls ** 2)


def _loss_grad(states, controls):
    return jax.grad(_loss, argnums=(0, 1))(states, controls)


def _flat(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WEIGHT_KEYS])


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _make_solve(solver):
    return make_central_path_diff(
        solver,
        slack_weight=_CFG["slack_weight"],
        target_kappa=_CFG["target_kappa"],
        include_ineq_hessian=True,
        conv_slack_weight=_CFG["conv_slack_weight"],
        kappa_anneal=_CFG["kappa_anneal"],
        kappa_anneal_start=_CFG["kappa_anneal_start"],
        kappa_anneal_factor=_CFG["kappa_anneal_factor"],
        linesearch=_CFG["linesearch"],
        max_sqp_iter=_CFG["max_sqp_iter"],
        tol=_CFG["tol"],
        jit_inner=_CFG["jit_inner"],
    )


def _fd_grad_weights(loss_of_w, weights, eps_seq=(1e-3, 3e-4, 1e-4), rtol=2e-3, atol=1e-7):
    """Convergence-checked central-diff gradient over a weights dict (plateau or flag)."""
    base = {k: np.array(weights[k], float) for k in WEIGHT_KEYS}
    grad, flagged = [], False
    for k in WEIGHT_KEYS:
        gk = np.zeros(base[k].size)
        for i in range(base[k].size):
            vals = []
            for eps in eps_seq:
                ap = base[k].copy().reshape(-1); ap[i] += eps
                am = base[k].copy().reshape(-1); am[i] -= eps
                wp = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}
                wp[k] = jnp.asarray(ap.reshape(base[k].shape))
                wm = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}
                wm[k] = jnp.asarray(am.reshape(base[k].shape))
                vals.append((loss_of_w(wp) - loss_of_w(wm)) / (2 * eps))
            chosen, ok = vals[-1], False
            for a, b in zip(vals[:-1], vals[1:]):
                if abs(a - b) <= rtol * abs(b) + atol:
                    chosen, ok = a, True; break
            flagged = flagged or (not ok); gk[i] = chosen
        grad.append(gk)
    return np.concatenate(grad), flagged


def _fd_grad_array(loss_of_x, x0, eps_seq=(1e-3, 3e-4, 1e-4), rtol=2e-3, atol=1e-7):
    """Convergence-checked central-diff gradient over a flat array argument."""
    base = np.array(x0, float)
    g = np.zeros(base.size)
    flagged = False
    for i in range(base.size):
        vals = []
        for eps in eps_seq:
            xp = base.copy().reshape(-1); xp[i] += eps
            xm = base.copy().reshape(-1); xm[i] -= eps
            vp = float(loss_of_x(jnp.asarray(xp.reshape(base.shape))))
            vm = float(loss_of_x(jnp.asarray(xm.reshape(base.shape))))
            vals.append((vp - vm) / (2 * eps))
        chosen, ok = vals[-1], False
        for a, b in zip(vals[:-1], vals[1:]):
            if abs(a - b) <= rtol * abs(b) + atol:
                chosen, ok = a, True; break
        flagged = flagged or (not ok); g[i] = chosen
    return g, flagged


def test_gate1_forward_unchanged():
    """solve(pp,w)[0] (states) == central_path_nlp_solve(...)['states'] (forward identical)."""
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    solve = _make_solve(solver)

    states_solve, _ = solve(pp, weights)
    res = central_path_nlp_solve(solver, pp, weights, **_CFG)
    # Same forward computation; differences are only GPU non-determinism (~1e-13 noise
    # floor, CLAUDE.md), far below any meaningful change.
    diff = float(jnp.max(jnp.abs(states_solve - res["states"])))
    print(f"[gate1] forward max abs diff = {diff:.3e}")
    assert diff < 1e-9, (
        f"custom_vjp forward must match central_path_nlp_solve (max abs diff {diff:.3e})"
    )


def test_gate2_weight_grad_matches_fd_and_nlp_grad():
    """jax.grad(solve) w.r.t. weights == convergence-checked FD == central_path_nlp_grad."""
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    solve = _make_solve(solver)

    g_ad = jax.grad(lambda w: _loss(*solve(pp, w)))(weights)
    g_ad_flat = _flat(g_ad)

    # (a) matches the existing explicit backward (refactor preserved dL_dweights).
    _res, g_nlp, _info = central_path_nlp_grad(
        solver, pp, weights, _loss_grad, include_ineq_hessian=True, **_CFG)
    g_nlp_flat = _flat(g_nlp)
    assert float(np.max(np.abs(g_ad_flat - g_nlp_flat))) < 1e-9, (
        f"custom_vjp weight grad != central_path_nlp_grad "
        f"(max abs diff {float(np.max(np.abs(g_ad_flat - g_nlp_flat))):.2e})"
    )

    # (b) matches convergence-checked FD over weights.
    def fwd_loss(w):
        return float(_loss(*solve(pp, w)))
    g_fd, flagged = _fd_grad_weights(fwd_loss, weights)
    assert not flagged, "weight-grad FD did not plateau"
    c, r = _cos(g_ad_flat, g_fd), _rel(g_ad_flat, g_fd)
    print(f"[gate2] weight grad: cos={c:.8f} rel={r:.3e} flagged={flagged}")
    assert c > 1 - 1e-4, f"weight grad AD vs FD cos={c}"
    assert r < 1e-2, f"weight grad AD vs FD rel={r}"


def test_gate3_x0_grad_matches_fd_SIGN():
    """THE SIGN GATE: jax.grad(solve) w.r.t. initial_state == convergence-checked FD."""
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    solve = _make_solve(solver)
    x0 = jnp.asarray(START)

    def loss_x0(xi):
        return _loss(*solve({**pp, "initial_state": xi}, weights))

    g_ad = np.asarray(jax.grad(loss_x0)(x0))
    g_fd, flagged = _fd_grad_array(loss_x0, x0)
    c, r = _cos(g_ad, g_fd), _rel(g_ad, g_fd)
    print(f"[gate3] x0 grad SIGN: cos={c:.8f} rel={r:.3e} flagged={flagged}")
    print(f"[gate3]   g_ad={g_ad}")
    print(f"[gate3]   g_fd={g_fd}")
    assert not flagged, "x0-grad FD did not plateau"
    assert c > 1 - 1e-4, (
        f"x0 grad AD vs FD cos={c} (if cos~-1, dL_dx_init sign is flipped in backward.py)"
    )
    assert r < 1e-2, f"x0 grad AD vs FD rel={r}"


def test_gate4_two_step_bptt_chained_x0_feedback():
    """2-step BPTT: x0->solve->u0->x1->solve->u1->x2, dL/dw through the dL/dx0 chain == FD."""
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    solve = _make_solve(solver)
    x0 = jnp.asarray(START)

    def simulate(x, u):
        return x + DT * (A_C @ x + B_C @ u)

    def bptt_loss(w):
        u0 = solve({**pp, "initial_state": x0}, w)[1][0]
        x1 = simulate(x0, u0)
        u1 = solve({**pp, "initial_state": x1}, w)[1][0]
        x2 = simulate(x1, u1)
        return (jnp.sum((x1[:2] - GOAL[:2]) ** 2) + jnp.sum((x2[:2] - GOAL[:2]) ** 2))

    g_ad = _flat(jax.grad(bptt_loss)(weights))
    g_fd, flagged = _fd_grad_weights(lambda w: float(bptt_loss(w)), weights)
    c, r = _cos(g_ad, g_fd), _rel(g_ad, g_fd)
    print(f"[gate4] 2-step BPTT: cos={c:.8f} rel={r:.3e} flagged={flagged}")
    print(f"[gate4]   g_ad={g_ad}")
    print(f"[gate4]   g_fd={g_fd}")
    assert not flagged, "BPTT-grad FD did not plateau"
    assert c > 1 - 1e-4, f"BPTT grad AD vs FD cos={c}"
    assert r < 1e-2, f"BPTT grad AD vs FD rel={r}"
