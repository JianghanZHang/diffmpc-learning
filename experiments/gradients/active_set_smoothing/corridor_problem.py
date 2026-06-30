"""Convex coupled-linear 'corridor' OCP — the Phase-2 stepping stone (no inequality Hessian needed).

Mirrors `external/turbompc/turbompc/problems/obstacle_avoidance.py` EXACTLY, but appends **linear**
half-plane state constraints  a_i^T p - b_i <= 0  instead of the nonlinear obstacle. Linear rows have
grad^2 g = 0, so the existing central-path backward (D + G1^T diag(W) G1) is exact — this validates the
sweep / conditioning / tol-band harness before we implement the obstacle's curvature term.

When several walls meet at a corner and the tracking goal sits just outside it, raising the position-
tracking weight presses the terminal position into the corner where >=2 walls activate simultaneously
-> a COUPLED near-active set (the conditioning stressor box constraints don't produce).

The obstacle swap later: replace `a^T p - b` with `1 - ||p - c|| / r` and add grad^2 g (Phase 3).
"""
from typing import Any, Dict, Tuple

import jax.numpy as jnp
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    make_slack_problem,
)


class OptimalControlProblemCorridor(OptimalControlProblem):
    """Quadratic tracking OCP with linear half-plane (corridor) state constraints."""

    def step_inequality_constraints(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Base (box) constraints with linear corridor walls  a_i^T p - b_i <= 0  appended."""
        g, g_l, g_u = super().step_inequality_constraints(state, control, params)
        dim = self.params["corridor_dimension"]
        position = state[:dim]
        A = jnp.asarray(params["corridor_A"])          # (n_walls, dim)
        b = jnp.asarray(params["corridor_b"])          # (n_walls,)

        g_wall = A @ position - b                       # a^T p - b <= 0
        g_wall_l = -1e9 * jnp.ones_like(g_wall)
        g_wall_u = jnp.zeros_like(g_wall)
        if self.rescale_optimization_variables:
            _, _, _, _, state_diff, _ = self._get_rescaling_params(params)
            row_scale = 1.0 / jnp.mean(state_diff[:dim])
            g_wall = g_wall * row_scale
            g_wall_l = g_wall_l * row_scale
            g_wall_u = g_wall_u * row_scale

        g = jnp.concatenate([g, g_wall])
        g_l = jnp.concatenate([g_l, g_wall_l])
        g_u = jnp.concatenate([g_u, g_wall_u])
        return (g, g_l, g_u)


OptimalControlProblemCorridorSlack = make_slack_problem(OptimalControlProblemCorridor)

__all__ = ["OptimalControlProblemCorridor", "OptimalControlProblemCorridorSlack"]
