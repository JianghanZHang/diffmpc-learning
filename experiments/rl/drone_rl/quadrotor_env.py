"""Nonlinear quadrotor environment for Diff-WMPC training.

The genuinely NONLINEAR sibling of ``drone_env.py``: a full 13-state quadrotor
(``QuadrotorDynamics``) instead of the 6-state double integrator.  Provides the
OCP problem params, a closed-loop simulate step, a (quaternion-aware) task loss,
and an obstacle-margin helper.  Exposes the SAME public API as ``drone_env`` so the
dynamics-agnostic pipeline (``mpc_layer``, ``policy``, ``optimizer``,
``gradient_modes``, ``train``) is reused unchanged.

State (nx=13):  [px,py,pz,  vx,vy,vz,  q0,q1,q2,q3,  wx,wy,wz]
                 pos(3)      vel(3)     quat(4=[w,x,y,z])  body-rate(3)
Control (nu=4): [thrust, tau_x, tau_y, tau_z]   (hover thrust = m*g)

Task: fly from a displaced START back to hover at the ORIGIN while avoiding ONE
circular obstacle.  The obstacle is a 2-D xy-CYLINDER (constraint on ``state[:2]``,
altitude-independent) placed on the START->GOAL line, so the realized path must
detour AROUND it (it cannot be escaped by climbing in z) — the regime where real
active-set switches can finally trigger.

Why a quaternion-aware loss: a naive ``||q - q_ref||^2`` is wrong because q and -q
are the SAME rotation (double cover) and the raw difference is not an SO(3) metric.
We regulate position + velocity + body-rate and use the sign-invariant attitude
error ``1 - (q . q_ref)^2 / ||q||^2`` (q_ref = identity ⇒ ``1 - q0^2/||q||^2``),
which is smooth, sign-invariant, and robust to the small quaternion-norm drift that
RK4 integration introduces.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from turbompc.dynamics.quadrotor_dynamics import (
    QuadrotorDynamics,
    quadrotor_parameters as QR_PARAMS,
    quadrotor_state_dot_parameters as QR_DYN,
)
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

NX: int = 13
NU: int = 4

# ---------------------------------------------------------------------------
# Hover reference (the GOAL) and nominal cost weights
# ---------------------------------------------------------------------------

HOVER_THRUST: float = float(QR_DYN["mass"]) * 9.81  # m * g

# hover at the origin, identity quaternion, zero velocity / body-rate
GOAL: jnp.ndarray = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=jnp.float64,
)
REF_CONTROL: jnp.ndarray = jnp.array(
    [HOVER_THRUST, 0.0, 0.0, 0.0], dtype=jnp.float64
)

# Default Q (per-state) / R (per-control) cost weights — pos,vel,quat,omega / thrust,torques.
# These are the weights the zero-init policy reproduces (weights = default * exp(NN(state))).
#
# Deliberately POOR prior: LOW position weight (0.5). The zero-init policy therefore starts as a
# weakly-goal-seeking controller that does NOT reach the goal in the eval window (closed-loop eval
# ~85, goaldist ~1.2 — it drifts up to the obstacle and sits GRAZING it). The NN must learn to
# RAISE the position weight to push past the obstacle and reach the goal (eval -> ~33). This is the
# drone-analog learnable regime: the well-tuned default sits at the ~33.6 eval FLOOR (nothing to
# learn -> flat), whereas this poor default leaves a clear ~85->33 gap to descend, all while the
# trajectory grazes the obstacle (real active-set switches). Verified by diag_weight_sensitivity.py
# (H=25: lowpos[0.5]=85 grazing/no-reach, balanced[10]=33.6 reach). [vel/quat/omega kept at the
# well-tuned 1/10/1 so only the goal-seeking strength is what the policy must learn.]
DEFAULT_Q: jnp.ndarray = jnp.array(
    [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    dtype=jnp.float64,
)
DEFAULT_R: jnp.ndarray = jnp.array([0.1, 0.1, 0.1, 0.1], dtype=jnp.float64)

# ---------------------------------------------------------------------------
# Problem geometry — START displaced from GOAL; obstacle blocks the direct path
# ---------------------------------------------------------------------------

# Start 1.5 m back along -x, at rest, level attitude.
START: jnp.ndarray = jnp.array(
    [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=jnp.float64,
)

# ONE xy-cylinder obstacle on the START->GOAL (x-axis) line.  Center offset slightly
# in +y to break the ±y detour symmetry.  Tuned (see verify_quadrotor.py) so the
# realized closed-loop path GRAZES the boundary (active-set-sensitive regime).
OBS_C: jnp.ndarray = jnp.array([-0.75, 0.05], dtype=jnp.float64)
OBS_R: float = 0.45

# ---------------------------------------------------------------------------
# MPC / discretization parameters
# ---------------------------------------------------------------------------

DT: float = 0.05          # seconds per step
HORIZON: int = 25         # planning horizon N (N*DT = 1.25 s lookahead)
DISCR_SCHEME: int = int(DiscretizationScheme.RUNGEKUTTA4)  # = 2
UMAX: float = 4.0         # symmetric control box; large ⇒ obstacle (not thrust) is the binding constraint

# Weight-dict keys (used by the MPC solver)
QK: str = "weights_penalization_reference_state_trajectory"
RK: str = "weights_penalization_control_squared"

# ---------------------------------------------------------------------------
# Internal dynamics singleton (do NOT export directly — use build_problem_params)
# ---------------------------------------------------------------------------

_DYNAMICS = QuadrotorDynamics(QR_PARAMS)

# Single-step dynamics params (mass scalar + inertia (9,)); state_dot reshapes inertia to (3,3).
_STATE_DOT_PARAMS: dict = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in QR_DYN.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_problem_params() -> tuple:
    """Return (dynamics, problem_params) for the single-obstacle quadrotor OCP.

    Mirrors ``build_quadrotor_problem`` (the validated hover-regulation template) but
    swaps in ``OptimalControlProblemObstacle`` fields and the START->GOAL geometry.

    Returns
    -------
    dynamics : QuadrotorDynamics
    pp : dict
        Problem-params dict suitable for OptimalControlProblemObstacle.  Key obstacle
        fields: obstacles_centers (N+1,1,2), obstacles_radii (1,), obstacles_dimension=2,
        use_slack_variables=False, discretization_scheme=RK4, rescale=False.
    """
    N = HORIZON
    pp: dict = {
        "horizon": N,
        "discretization_resolution": DT,
        "discretization_scheme": DISCR_SCHEME,  # RK4
        "initial_state": START,
        "initial_guess_final_state": GOAL,
        "reference_state_trajectory": jnp.tile(GOAL, (N + 1, 1)),
        "reference_control_trajectory": jnp.tile(REF_CONTROL, (N + 1, 1)),
        "penalize_control_reference": True,  # track hover thrust (control ref != 0)
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": REF_CONTROL,
        "state_rescaling_min": -jnp.ones((NX,)),
        "state_rescaling_max": jnp.ones((NX,)),
        "control_rescaling_min": -jnp.ones((NU,)),
        "control_rescaling_max": jnp.ones((NU,)),
        QK: DEFAULT_Q,
        "weights_penalization_final_state": jnp.zeros((NX,)),
        RK: DEFAULT_R,
        "weights_penalization_control_rate": jnp.zeros((NU,)),
        "state_min_bounds": -jnp.ones((NX,)) * 1.0e7,
        "state_max_bounds": jnp.ones((NX,)) * 1.0e7,
        "control_min_bounds": -jnp.ones((NU,)) * UMAX,
        "control_max_bounds": jnp.ones((NU,)) * UMAX,
        "dynamics_state_dot_params": {k: jnp.asarray(v, dtype=jnp.float64) for k, v in QR_DYN.items()},
        # ---- obstacle (one 2-D xy-cylinder, tiled over time steps) ----
        "obstacles_centers": jnp.tile(OBS_C[None, None, :], (N + 1, 1, 1)),  # (N+1, 1, 2)
        "obstacles_radii": jnp.array([OBS_R], dtype=jnp.float64),
        "obstacles_dimension": 2,
        "use_slack_variables": False,
        "slack_penalization_weight": 0.0,
    }
    return _DYNAMICS, pp


def simulate_step(
    dynamics: QuadrotorDynamics,
    x: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    """Advance one closed-loop step with the SAME model/scheme/dt as the MPC (RK4).

    Plan first-step == realized step (zero model mismatch), so the per-step plan loss
    and realized loss agree in value; the gradient ESTIMATOR is what differs across
    variants.

    Parameters
    ----------
    dynamics : QuadrotorDynamics  (must be the object returned by build_problem_params).
    x : (NX,) current state.
    u : (NU,) control to apply.

    Returns
    -------
    x_next : (NX,)
    """
    return predict_next_state(
        dynamics,
        DT,
        DiscretizationScheme.RUNGEKUTTA4,
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
    w_att: float = 0.5,
    w_omega: float = 0.05,
    w_u: float = 1e-3,
) -> jnp.ndarray:
    """Quaternion-aware trajectory task loss: reach hover at GOAL, level + still.

    Terms (summed over the supplied trajectory slice):
      position  : w_p * ||p - p_goal||^2
      velocity  : w_v * ||v||^2
      attitude  : w_att * (1 - q0^2/||q||^2)   — sign-invariant SO(3) error to identity
      body-rate : w_omega * ||omega||^2
      control   : w_u * ||u - u_hover||^2       — penalize deviation from hover thrust

    The attitude term uses the normalized ``1 - (q . q_ref)^2`` metric (q_ref = identity
    ⇒ ``1 - q0^2/||q||^2``): smooth, invariant to the q/-q double cover, and robust to
    RK4 quaternion-norm drift.  We deliberately do NOT penalize the raw quaternion
    difference.

    Parameters
    ----------
    states   : (*batch, NX) state trajectory (or batch thereof).
    controls : (*batch, NU) control trajectory.

    Returns
    -------
    loss : scalar jnp.ndarray
    """
    pos_err = states[..., 0:3] - GOAL[0:3]
    vel = states[..., 3:6]
    quat = states[..., 6:10]            # [q0,q1,q2,q3] = [w,x,y,z]
    omega = states[..., 10:13]
    q_sq = jnp.sum(quat ** 2, axis=-1)  # ||q||^2 (RK4 drift ⇒ not exactly 1)
    att_err = 1.0 - quat[..., 0] ** 2 / (q_sq + 1e-12)  # 1 - cos^2(angle to identity)
    du = controls - REF_CONTROL
    return (
        w_p * jnp.sum(pos_err ** 2)
        + w_v * jnp.sum(vel ** 2)
        + w_att * jnp.sum(att_err)
        + w_omega * jnp.sum(omega ** 2)
        + w_u * jnp.sum(du ** 2)
    )


def obs_margin(x: jnp.ndarray) -> jnp.ndarray:
    """Obstacle margin (matches the OCP obstacle constraint exactly):

        h = 1 - ||x[:2] - OBS_C|| / (OBS_R + 1e-3)

    Parameters
    ----------
    x : (NX,) or (*, NX) state(s).

    Returns
    -------
    margin : > 0  → infeasible (inside the xy-cylinder); <= 0 → feasible (outside).
    """
    pos_xy = x[..., :2]
    dist = jnp.linalg.norm(pos_xy - OBS_C, axis=-1)
    return 1.0 - dist / (OBS_R + 1e-3)


def goal_dist(x: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance from the (3-D) position to the GOAL position (origin)."""
    return jnp.linalg.norm(x[..., :3] - GOAL[:3], axis=-1)


def sample_x0(key: jnp.ndarray, scale: float = 0.02) -> jnp.ndarray:
    """A perturbed START for an episode reset: ``START + N(0, scale^2)`` with the
    quaternion (indices 6:10) RENORMALIZED so the start is always a valid unit
    quaternion (the dynamics divide by ||q||, but the cost/loss should see unit q)."""
    x = START + jax.random.normal(key, START.shape, dtype=jnp.float64) * scale
    q = x[6:10]
    q = q / jnp.linalg.norm(q)
    return x.at[6:10].set(q)
