"""Drone environment for Diff-WMPC training.

Single-obstacle 6-state double-integrator drone (with quadratic drag).
Provides the OCP problem params, a closed-loop simulate step, a task loss,
and an obstacle-margin helper.  All public names are imported by later tasks.

State:   [px, py, pz, vx, vy, vz]
Control: [Fx, Fy, Fz]
Obstacle: ONE circular obstacle (in the xy-plane) placed on the START->GOAL
          path so the constraint is ACTIVE / near-grazing.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from turbompc.dynamics.drone_dynamics import (
    DroneDynamics,
    drone_parameters,
    drone_state_dot_parameters,
)
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.utils.load_params import load_problem_params

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

NX: int = 6
NU: int = 3

# ---------------------------------------------------------------------------
# Problem geometry
# ---------------------------------------------------------------------------

# Drone starts slightly off-axis and must reach the origin.
START: jnp.ndarray = jnp.array([-1.9, 0.05, 0.2, 0.0, 0.0, 0.0], dtype=jnp.float64)
GOAL: jnp.ndarray = jnp.zeros(6, dtype=jnp.float64)

# ONE obstacle placed near the midpoint of the START->(0,0) xy-segment so the
# optimal path is forced to deviate.  Adjust OBS_C / OBS_R here if the test
# reports the obstacle is not engaged.
OBS_C: jnp.ndarray = jnp.array([-0.95, 0.025], dtype=jnp.float64)
OBS_R: float = 0.3

# ---------------------------------------------------------------------------
# MPC / discretization parameters
# ---------------------------------------------------------------------------

DT: float = 1.0          # seconds per step (matches drone.yaml default)
HORIZON: int = 50        # planning horizon N
DISCR_SCHEME: int = 0    # DiscretizationScheme.EULER = 0

# Weight-dict keys (used by the MPC solver)
QK: str = "weights_penalization_reference_state_trajectory"
RK: str = "weights_penalization_control_squared"

# ---------------------------------------------------------------------------
# Internal dynamics singleton (do NOT export directly — use build_problem_params)
# ---------------------------------------------------------------------------

_DYNAMICS = DroneDynamics(drone_parameters)

# Single-step dynamics params extracted from drone_state_dot_parameters (scalars)
_STATE_DOT_PARAMS: dict = {
    k: jnp.asarray(v[0]) for k, v in drone_state_dot_parameters.items()
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_problem_params() -> tuple:
    """Return (dynamics, problem_params) for the single-obstacle drone OCP.

    Returns
    -------
    dynamics : DroneDynamics
        The drone dynamics object (reused by simulate_step).
    pp : dict
        Problem-params dict suitable for OptimalControlProblemObstacle.
        Key fields:
          - obstacles_centers   shape (HORIZON+1, 1, 2)  tiled from OBS_C
          - obstacles_radii     shape (1,)
          - obstacles_dimension = 2
          - use_slack_variables = False
          - discretization_scheme = DISCR_SCHEME (EULER)
          - rescale_optimization_variables = False
    """
    base = dict(load_problem_params("drone.yaml"))

    # Horizon & dt
    base["horizon"] = HORIZON
    base["discretization_resolution"] = DT
    base["discretization_scheme"] = DISCR_SCHEME  # EULER
    # constrain_initial_control only makes sense for the implicit scheme
    base["constrain_initial_control"] = False

    N = HORIZON
    nx = _DYNAMICS.num_states
    nu = _DYNAMICS.num_controls

    # Reference trajectories (regulate to GOAL=0)
    base["reference_state_trajectory"] = jnp.zeros((N + 1, nx), dtype=jnp.float64)
    base["reference_control_trajectory"] = jnp.zeros((N + 1, nu), dtype=jnp.float64)

    # Initial state / guess
    base["initial_state"] = START
    base["initial_guess_final_state"] = GOAL

    # Obstacle: one obstacle, tiled over time steps
    # obstacles_centers shape: (N+1, n_obs, dim)
    obs_centers_tiled = jnp.tile(
        OBS_C[None, None, :], (N + 1, 1, 1)
    )  # (N+1, 1, 2)
    base["obstacles_centers"] = obs_centers_tiled
    base["obstacles_radii"] = jnp.array([OBS_R], dtype=jnp.float64)
    base["obstacles_dimension"] = 2

    # No slack, no rescaling
    base["use_slack_variables"] = False
    base["slack_penalization_weight"] = 0.0
    base["rescale_optimization_variables"] = False

    # Dynamics state-dot params: shape (N, dim) for time-varying vectors.
    # drone_state_dot_parameters values are 1D arrays (e.g. jnp.ones(1));
    # tile them as (N, 1) so the solver doesn't raise "ambiguous 1D param" errors.
    base["dynamics_state_dot_params"] = {
        k: jnp.tile(jnp.asarray(v, dtype=jnp.float64)[None, :], (N, 1))
        for k, v in drone_state_dot_parameters.items()
    }

    # Control bounds from yaml are already present; keep them
    return _DYNAMICS, base


def simulate_step(
    dynamics: DroneDynamics,
    x: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    """Advance one closed-loop step using the SAME model/scheme as the MPC.

    Uses EULER integration with DT and the same drone_state_dot_parameters so
    that the MPC plan's first transition equals the realized step (zero model
    mismatch).

    Parameters
    ----------
    dynamics : DroneDynamics
        Must be the same object returned by build_problem_params().
    x : jnp.ndarray, shape (NX,)
        Current state.
    u : jnp.ndarray, shape (NU,)
        Control to apply.

    Returns
    -------
    x_next : jnp.ndarray, shape (NX,)
    """
    return predict_next_state(
        dynamics,
        DT,
        DiscretizationScheme.EULER,
        _STATE_DOT_PARAMS,
        x,
        u,
    )


def task_loss(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    *,
    w_p: float = 1.0,
    w_v: float = 0.1,
    w_u: float = 1e-3,
) -> jnp.ndarray:
    """Trajectory task loss: reach GOAL position, damp velocity, control effort.

    Parameters
    ----------
    states : jnp.ndarray, shape (*batch, NX)
        State trajectory (or batch thereof).
    controls : jnp.ndarray, shape (*batch, NU)
        Control trajectory (or batch thereof).
    w_p, w_v, w_u : float
        Weights for position, velocity, and control terms.

    Returns
    -------
    loss : scalar jnp.ndarray
    """
    pos_err = states[..., :3] - GOAL[:3]
    vel = states[..., 3:6]
    return (
        w_p * jnp.sum(pos_err ** 2)
        + w_v * jnp.sum(vel ** 2)
        + w_u * jnp.sum(controls ** 2)
    )


def obs_margin(x: jnp.ndarray) -> jnp.ndarray:
    """Obstacle margin: positive means INSIDE the obstacle (constraint violated).

    Matches the obstacle_avoidance.py constraint exactly:
        h = 1 - ||x[:2] - OBS_C|| / (OBS_R + 1e-3)

    Parameters
    ----------
    x : jnp.ndarray, shape (NX,) or (*, NX)
        State or batch of states.

    Returns
    -------
    margin : scalar (or batch of scalars)
        > 0  → infeasible (inside obstacle)
        <= 0 → feasible (outside)
    """
    pos_xy = x[..., :2]
    dist = jnp.linalg.norm(pos_xy - OBS_C, axis=-1)
    return 1.0 - dist / (OBS_R + 1e-3)


def goal_dist(x: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance from the (xy) position to the GOAL position (origin)."""
    return jnp.linalg.norm(x[..., :2] - GOAL[:2], axis=-1)


def sample_x0(key: jnp.ndarray, scale: float = 0.02) -> jnp.ndarray:
    """A perturbed START for an episode reset: ``START + N(0, scale^2)``."""
    return START + jax.random.normal(key, START.shape, dtype=jnp.float64) * scale
