"""Forward-speed parity: the accelerated central-path ADMM converges in the same number
of iterations as TurboMPC's ADMM on the SAME one-sided QP.

`solve_qp_central_path` is a faithful port of TurboMPC's `_solve_jax_loop` (over-relaxation
alpha=1.6, OSQP-style adaptive rho with Schur rebuild, residual-based termination) whose only
difference is the elastic-retraction z-update. As kappa->0 the retraction equals the box/slack
projection, so the iteration counts should match (was ~100x slower before the port).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path

from turbompc.solvers.qp_utils import ZShape
from turbompc.solvers.admm.admm import ADMMSolver
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend, AdmmBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver

from benchmark_cartpole_coupling import build_cartpole_problem
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params

NX, NU = 4, 1
_POLE_DOWN = jnp.array([0.0, 0.0, jnp.pi, 0.0])
_GAMMA = 1.0e2
_KAPPA = 1.0e-6
_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}


def _cartpole_one_sided_qp(umax, slack_weight):
    dynamics, pp = build_cartpole_problem(horizon=25, umax=umax, dt=0.04)
    pp["initial_state"] = _POLE_DOWN
    ocp = OptimalControlProblem(dynamics=dynamics, params=pp)
    sp = load_solver_params("turbompc.yaml")
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI)
    ig = solver.initial_guess(pp)
    return to_one_sided(solver._build_qp_data(ig.states, ig.controls, pp), slack_weight=slack_weight)


def _turbompc_admm(qp1, N, tol):
    """TurboMPC's box/slack ADMM on the same one-sided QP (the iteration reference)."""
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)
    ref = ADMMSolver(
        zshape=ZShape(horizon=N, num_states=NX, num_controls=NU), schur_solver=schur,
        pcg_params=_PCG, sigma=1e-6, max_iter=100000, eps_abs=tol, eps_rel=tol,
        rho_f_factor=1000.0, admm_backend=AdmmBackend.JAX_LOOP, use_slack=True,
        adapt_rho_every=25, check_termination_every=25, adaptive_rho_tolerance=5.0)
    (states, controls), stats, _ = ref.solve(qp1, rho_bar=0.1, alpha=1.6, slack_weight=_GAMMA)
    return int(stats.num_iter), jnp.concatenate([states, controls], axis=-1)


def test_central_path_admm_iteration_parity_with_turbompc():
    qp1 = _cartpole_one_sided_qp(umax=2.0, slack_weight=_GAMMA)
    N = qp1.cost.D.shape[0] - 1
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)

    for tol in (1e-4, 1e-7):
        x_cp, _, info = solve_qp_central_path(
            qp1, schur, target_kappa=_KAPPA, slack_weight=_GAMMA, rho_bar=0.1,
            max_iter=100000, tol=tol)
        it_tm, x_tm = _turbompc_admm(qp1, N, tol)
        it_cp = int(info["iters"])

        # converged to the residual tolerance (the fixed convergence check)
        assert float(info["prim_res"]) < 10 * tol, f"tol={tol}: prim_res={float(info['prim_res'])}"
        assert float(info["dual_res"]) < 10 * tol, f"tol={tol}: dual_res={float(info['dual_res'])}"
        # iteration parity: same magnitude as TurboMPC (was ~100x before the port)
        assert it_cp <= 2 * it_tm + 10, f"tol={tol}: CP {it_cp} iters vs TurboMPC {it_tm}"
        # same solution up to the O(kappa) barrier bias (central path vs soft box)
        assert float(jnp.max(jnp.abs(x_cp - x_tm))) < 1e-2, f"tol={tol}: solution mismatch"
