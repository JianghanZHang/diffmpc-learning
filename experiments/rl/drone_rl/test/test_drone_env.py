"""TDD verification test for drone_env.py.

Run with:
    cd /home/jianghan/Workspace/diffmpc-learning
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/diffmpc2 /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \
        -m pytest experiments/rl/drone_rl/test/test_drone_env.py -v -s
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import sys
import os

# Make drone_env importable (same-dir) regardless of pytest import mode / folder location.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # drone_rl/

import jax.numpy as jnp
import numpy as np
import pytest

from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_solver_params

# Import the module under test (same-dir)
from env.drone_env import (
    NX,
    NU,
    START,
    GOAL,
    OBS_C,
    OBS_R,
    DT,
    HORIZON,
    DISCR_SCHEME,
    QK,
    RK,
    build_problem_params,
    simulate_step,
    task_loss,
    obs_margin,
)


# ---------------------------------------------------------------------------
# Solver factory (mirrors test_inequality_hessian.py pattern)
# ---------------------------------------------------------------------------

def _build_solver(pp: dict, dyn):
    """Build TurboMPCSolver with ADMM_FUSED_CUDSS / DIRECT_CUDSS_FFI."""
    ocp = OptimalControlProblemObstacle(dynamics=dyn, params=pp)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 50
    sp["tol_convergence"] = 1e-5
    sp["linesearch"] = True
    sp["linesearch_alphas"] = [0.1, 0.3, 0.7, 1.0]
    solver = TurboMPCSolver(
        program=ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True,
    )
    return solver, sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_and_solution():
    """Build params, solve once, return (dyn, pp, solver, sol)."""
    dyn, pp = build_problem_params()
    solver, _ = _build_solver(pp, dyn)
    ig = solver.initial_guess(pp)
    weights = {QK: pp[QK], RK: pp[RK]}
    sol = solver.solve(ig, pp, weights)
    return dyn, pp, solver, sol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_problem_params_keys():
    """build_problem_params() returns a valid pp with expected obstacle keys."""
    dyn, pp = build_problem_params()

    assert "obstacles_centers" in pp, "Missing obstacles_centers"
    assert "obstacles_radii" in pp, "Missing obstacles_radii"
    assert "obstacles_dimension" in pp, "Missing obstacles_dimension"
    assert pp["use_slack_variables"] is False, "Expected use_slack_variables=False"
    assert int(pp["discretization_scheme"]) == DISCR_SCHEME, \
        f"Expected EULER (0), got {pp['discretization_scheme']}"
    assert pp["rescale_optimization_variables"] is False, \
        "Expected rescale_optimization_variables=False"

    obs_c = jnp.asarray(pp["obstacles_centers"])
    assert obs_c.ndim == 3, f"Expected (N+1,1,2) obstacles_centers, got shape {obs_c.shape}"
    assert obs_c.shape == (HORIZON + 1, 1, 2), \
        f"obstacles_centers shape mismatch: {obs_c.shape}"

    print(f"\n[build] NX={NX}, NU={NU}, HORIZON={HORIZON}, DT={DT}")
    print(f"[build] OBS_C={np.array(OBS_C)}, OBS_R={OBS_R}")
    print(f"[build] pp keys: {sorted(pp.keys())}")


def test_forward_converges(env_and_solution):
    """Forward solve converges with convergence_error < 1e-5."""
    _, _, _, sol = env_and_solution
    conv_err = float(sol.convergence_error)
    print(f"\n[forward] convergence_error = {conv_err:.3e}")
    assert conv_err < 1e-5, f"Solver did not converge: convergence_error={conv_err:.3e}"


def test_obstacle_engaged(env_and_solution):
    """Obstacle is engaged: trajectory gets close to / grazes the boundary."""
    _, _, _, sol = env_and_solution
    states = np.array(sol.states)  # (N+1, NX)

    # Compute per-step margins and distances to obstacle center
    margins = []
    dists = []
    for k, x in enumerate(states):
        m = float(obs_margin(jnp.array(x)))
        d = float(np.linalg.norm(x[:2] - np.array(OBS_C)))
        margins.append(m)
        dists.append(d)
        print(f"  step {k:3d}: dist={d:.4f}, margin={m:.4f}")

    min_margin = min(margins)
    max_margin = max(margins)
    min_dist = min(dists)
    print(f"\n[obstacle] min_margin={min_margin:.4f}, max_margin={max_margin:.4f}")
    print(f"[obstacle] min_dist_to_center={min_dist:.4f}, OBS_R={OBS_R}")

    # Engagement check: max_margin is the CLOSEST approach to the obstacle boundary.
    # obs_margin = 0 means right at the boundary; obs_margin > 0 means inside.
    # max_margin > -0.2 → the path gets within 20% of OBS_R from the boundary.
    # max_margin <= 0.05 → hard constraint not grossly violated (no infeasibility).
    assert max_margin > -0.2, \
        f"Obstacle not engaged (path clears it by a lot): max_margin={max_margin:.4f}"
    assert max_margin <= 0.05, \
        f"Hard obstacle constraint grossly violated: max_margin={max_margin:.4f} > 0.05"


def test_simulate_step_matches_plan(env_and_solution):
    """simulate_step(x0, u0) lands within ~1e-6 of plan's states[1]."""
    dyn, pp, _, sol = env_and_solution
    x0 = jnp.array(START, dtype=jnp.float64)
    u0 = jnp.array(sol.controls[0], dtype=jnp.float64)

    x_next_sim = np.array(simulate_step(dyn, x0, u0))
    x_next_plan = np.array(sol.states[1])

    err = float(np.linalg.norm(x_next_sim - x_next_plan))
    print(f"\n[simulate_step] sim   states[1] = {x_next_sim}")
    print(f"[simulate_step] plan  states[1] = {x_next_plan}")
    print(f"[simulate_step] |error|         = {err:.3e}")

    assert err < 1e-6, \
        f"simulate_step vs MPC plan mismatch: |error|={err:.3e} >= 1e-6"


def test_task_loss_finite_and_differentiable():
    """task_loss returns a finite positive scalar and is JAX-differentiable."""
    # Use a small dummy trajectory (not the solved one, to keep this test standalone)
    states = jnp.ones((HORIZON + 1, NX), dtype=jnp.float64) * 0.1
    controls = jnp.ones((HORIZON, NU), dtype=jnp.float64) * 0.01

    loss = task_loss(states, controls)
    loss_val = float(loss)
    print(f"\n[task_loss] loss = {loss_val:.6f}")
    assert jnp.isfinite(loss), f"task_loss not finite: {loss_val}"
    assert loss_val > 0.0, f"task_loss not positive: {loss_val}"

    # Differentiability check via jax.grad
    def loss_fn(s):
        return task_loss(s, controls)

    grad = jax.grad(loss_fn)(states)
    grad_norm = float(jnp.linalg.norm(grad))
    print(f"[task_loss] ||grad w.r.t. states|| = {grad_norm:.6f}")
    assert jnp.all(jnp.isfinite(grad)), "task_loss gradient contains non-finite values"
    assert grad_norm > 0.0, "task_loss gradient is zero"
