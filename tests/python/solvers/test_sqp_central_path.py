"""NLP-KKT convergence: SQP with the central-path ADMM as the inner QP solver.

Drives TurboMPC's first-order NLP-KKT residual (`_compute_first_order_convergence_error`)
below tolerance on the nonlinear pole-down cartpole, with `solve_qp_central_path` swapped in
as the inner QP solver, for BOTH the hard-box NLP and the soft-box NLP. The hard solution is
checked against `TurboMPCSolver.solve` (no slack); the soft solution against the verified
two-sided quadratic-slack ADMM reference at the same gamma.
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diffmpc_learning.solvers.sqp import sqp_central_path
from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path  # noqa: F401

from turbompc.solvers.qp_utils import ZShape
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend, AdmmBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver

from benchmark_cartpole_coupling import build_cartpole_problem
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params
from turbompc.solvers.admm.admm import ADMMSolver

NX, NU = 4, 1
_POLE_DOWN = jnp.array([0.0, 0.0, jnp.pi, 0.0])
_GAMMA = 1.0e2          # soft-box slack penalty (matches the central-path equivalence tests)
_HARD_GAMMA = 1.0e8     # huge slack penalty => hard box
_UMAX = 2.0
_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}


def _build_solver(linesearch):
    """Cartpole TurboMPCSolver (no program-level slack) with cuDSS backends."""
    dynamics, pp = build_cartpole_problem(horizon=25, umax=_UMAX, dt=0.04)
    pp["initial_state"] = _POLE_DOWN
    ocp = OptimalControlProblem(dynamics=dynamics, params=pp)
    sp = load_solver_params("turbompc.yaml")
    sp = dict(sp)
    sp["linesearch"] = bool(linesearch)
    sp["num_sqp_iteration_max"] = 60
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    )
    return solver, pp


def _soft_reference(solver, pp, slack_weight):
    """Two-sided quadratic-slack soft NLP reference via the stock SQP loop.

    Runs the same SQP linearization as TurboMPC but with the verified two-sided soft ADMM
    inner solve (use_slack=True) so the slack penalty gamma is actually applied on the
    JAX_LOOP path. Relinearizes per SQP iteration with line search, matching `solver.solve`.
    """
    program = solver.program
    nx, nu, N = NX, NU, program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    ref = ADMMSolver(
        zshape=ZShape(horizon=N, num_states=nx, num_controls=nu),
        schur_solver=schur, pcg_params=_PCG,
        sigma=1e-6, max_iter=50000, eps_abs=1e-11, eps_rel=1e-9,
        rho_f_factor=1000.0, admm_backend=AdmmBackend.JAX_LOOP, use_slack=True,
    )
    states, controls = program.initial_guess(pp)
    for _ in range(60):
        qp = solver._build_qp_data(states, controls, pp)
        qp1 = to_one_sided(qp, slack_weight=slack_weight)
        (s_new, c_new), _, _ = ref.solve(qp1, rho_bar=0.1, slack_weight=slack_weight)
        step = max(float(jnp.max(jnp.abs(s_new - states))),
                   float(jnp.max(jnp.abs(c_new - controls))))
        states, controls = s_new, c_new
        if step < 1e-9:
            break
    return states, controls


def _rel_linf(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (1e-12 + jnp.max(jnp.abs(b))))


def test_nlp_kkt_converges_hard_box():
    solver, pp = _build_solver(linesearch=True)
    ref = solver.solve(solver.initial_guess(pp), pp)

    # Full Newton steps + a short kappa-anneal: the central-path inner solve produces an
    # exact, feasible QP step so full steps converge quadratically (linesearch=False; the
    # merit-based line search's slack mismatch stalls the step near the soft solution -- see
    # report). The anneal drives the inner barrier kappa 1e-3 -> 1e-7 below the O(kappa) floor.
    out = sqp_central_path(
        solver, pp,
        slack_weight=_HARD_GAMMA, target_kappa=1e-7,
        conv_slack_weight=None, linesearch=False,
        max_sqp_iter=60, tol=1e-5,
        kappa_anneal=True, kappa_anneal_start=1e-3, kappa_anneal_factor=0.1,
    )
    ch = out["conv_history"]
    print("HARD conv_history:", np.array2string(np.asarray(ch), precision=3))
    print(f"HARD final_conv={out['final_conv']:.3e}  num_iter={out['num_iter']}")

    assert out["final_conv"] < 1e-4, f"NLP-KKT did not reach 1e-4: {out['final_conv']:.3e}"
    assert float(ch[-1]) <= float(ch[0]) / 10.0, "conv did not decrease by >=10x"

    rel_s = _rel_linf(out["states"], ref.states)
    rel_c = _rel_linf(out["controls"], ref.controls)
    print(f"HARD rel-linf states={rel_s:.3e} controls={rel_c:.3e}")
    assert rel_s < 1e-2 and rel_c < 1e-2


def test_nlp_kkt_converges_soft_box():
    solver, pp = _build_solver(linesearch=True)
    states_ref, controls_ref = _soft_reference(solver, pp, _GAMMA)

    out = sqp_central_path(
        solver, pp,
        slack_weight=_GAMMA, target_kappa=1e-7,
        conv_slack_weight=_GAMMA, linesearch=False,
        max_sqp_iter=60, tol=1e-5,
        kappa_anneal=True, kappa_anneal_start=1e-3, kappa_anneal_factor=0.1,
    )
    ch = out["conv_history"]
    print("SOFT conv_history:", np.array2string(np.asarray(ch), precision=3))
    print(f"SOFT final_conv={out['final_conv']:.3e}  num_iter={out['num_iter']}")

    assert out["final_conv"] < 1e-4, f"NLP-KKT did not reach 1e-4: {out['final_conv']:.3e}"
    assert float(ch[-1]) <= float(ch[0]) / 10.0, "conv did not decrease by >=10x"

    rel_s = _rel_linf(out["states"], states_ref)
    rel_c = _rel_linf(out["controls"], controls_ref)
    print(f"SOFT rel-linf states={rel_s:.3e} controls={rel_c:.3e}")
    assert rel_s < 1e-2 and rel_c < 1e-2
