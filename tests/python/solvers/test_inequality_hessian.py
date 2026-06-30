"""Inequality-constraint Lagrangian Hessian (mu^T grad^2 g) in the LOCAL central-path backward.

Mirror of the turbompc test (external/turbompc/tests/.../test_inequality_hessian.py) for the
project's central-path NLP backward: `central_path_nlp_grad` now calls turbompc's
`get_inequality_lagrangian_hessian` (backward.py) and folds mu^T grad^2 g into the exact
Lagrangian Hessian, alongside the dynamics Hessian.

Problem: 2-D double integrator (LINEAR dynamics -> dynamics Hessian = 0) + ONE circular
obstacle `1 - ||p-c||/r <= 0`, off-axis goal (no degenerate direction), obstacle firmly
active. With the inequality Hessian (default) the relaxed AD gradient matches the
convergence-checked FD of the SAME relaxed forward; the `include_ineq_hessian=False`
ablation is materially worse -> the term is load-bearing for nonlinear constraints.

cuDSS Schur backend (central-path forward/backward use CUDSS_FFI internally). Run with the
cuDSS env (LD_LIBRARY_PATH + XLA_PYTHON_CLIENT_PREALLOCATE=false) and the .venv-cudss python.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diffmpc_learning.solvers.backward import central_path_nlp_solve, central_path_nlp_grad

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
WEIGHT_KEYS = [QK, RK]

_CFG = dict(slack_weight=1e4, target_kappa=1e-5, conv_slack_weight=1e4,
            kappa_anneal=True, kappa_anneal_start=1e-2, kappa_anneal_factor=0.3,
            linesearch=False, max_sqp_iter=60, tol=1e-6, jit_inner=True)


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
    # NOTE: central_path_nlp_grad does NOT use the TurboMPCSolver's backends — it runs its
    # own central-path forward (sqp_central_path, cuDSS Schur internally) and reduced-KKT
    # backward (solve_backward_kkt). These backends only feed _build_qp_data; set both to
    # cuDSS for consistency with the turbompc test.
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    return solver, cons


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


def _fd_grad_weights(loss_of_w, weights, eps_seq=(1e-3, 3e-4, 1e-4), rtol=2e-3, atol=1e-7):
    base = {k: np.array(weights[k], float) for k in WEIGHT_KEYS}
    grad, flagged = [], False
    for k in WEIGHT_KEYS:
        gk = np.zeros(base[k].size)
        for i in range(base[k].size):
            vals = []
            for eps in eps_seq:
                ap = base[k].copy().reshape(-1); ap[i] += eps
                am = base[k].copy().reshape(-1); am[i] -= eps
                wp = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}; wp[k] = jnp.asarray(ap.reshape(base[k].shape))
                wm = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}; wm[k] = jnp.asarray(am.reshape(base[k].shape))
                vals.append((loss_of_w(wp) - loss_of_w(wm)) / (2 * eps))
            chosen, ok = vals[-1], False
            for a, b in zip(vals[:-1], vals[1:]):
                if abs(a - b) <= rtol * abs(b) + atol:
                    chosen, ok = a, True; break
            flagged = flagged or (not ok); gk[i] = chosen
        grad.append(gk)
    return np.concatenate(grad), flagged


def test_forward_converges_and_obstacle_active():
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    res = central_path_nlp_solve(solver, pp, weights, **_CFG)
    assert float(res["final_eq"]) < 1e-5
    assert float(res["final_ineq"]) < 1e-4
    P = np.asarray(res["states"])[:, :2]
    dist = np.linalg.norm(P - np.asarray(OBS_C), axis=1)
    assert dist.min() < OBS_R + 0.05, f"obstacle not engaged (min dist {dist.min():.3f})"


def test_central_path_inequality_hessian_matches_fd():
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    _, dL_ad, info = central_path_nlp_grad(solver, pp, weights, _loss_grad,
                                           include_ineq_hessian=True, **_CFG)
    assert float(info["prim_res"]) < 1e-6
    g_ad = _flat(dL_ad)

    def fwd_loss(w):
        r = central_path_nlp_solve(solver, pp, w, **_CFG)
        return float(_loss(r["states"], r["controls"]))
    g_fd, flagged = _fd_grad_weights(fwd_loss, weights)
    assert not flagged, "FD did not plateau"
    assert _cos(g_ad, g_fd) > 1 - 1e-4, f"AD vs FD cos={_cos(g_ad, g_fd)}"
    assert _rel(g_ad, g_fd) < 1e-2, f"AD vs FD rel_l2={_rel(g_ad, g_fd)}"


def test_central_path_without_inequality_hessian_is_worse():
    solver, pp = _build_obstacle_solver()
    weights = {k: pp[k] for k in WEIGHT_KEYS}
    _, dL_on, _ = central_path_nlp_grad(solver, pp, weights, _loss_grad,
                                        include_ineq_hessian=True, **_CFG)
    _, dL_off, _ = central_path_nlp_grad(solver, pp, weights, _loss_grad,
                                         include_ineq_hessian=False, **_CFG)

    def fwd_loss(w):
        r = central_path_nlp_solve(solver, pp, w, **_CFG)
        return float(_loss(r["states"], r["controls"]))
    g_fd, flagged = _fd_grad_weights(fwd_loss, weights)
    assert not flagged
    rel_on, rel_off = _rel(_flat(dL_on), g_fd), _rel(_flat(dL_off), g_fd)
    assert rel_on < 1e-2, f"with-Hessian rel_l2={rel_on}"
    assert rel_off > 3 * rel_on, f"ablation not separated: on={rel_on:.2e} off={rel_off:.2e}"
