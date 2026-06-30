"""Small MLP policy (state -> MPC cost-weight multipliers) for the drone Diff-WMPC pipeline.

Mirrors external/turbompc/examples/pointmass_rl/policy.py. Zero-init output layer ⇒ the
initial policy emits zero log-multipliers ⇒ the MPC starts from the hand-chosen DEFAULT weights.
"""
from __future__ import annotations
from typing import Dict
import jax
import jax.numpy as jnp

Policy = Dict[str, jax.Array]


def init_policy(rng: jax.Array, obs_dim: int = 6, hidden: int = 64, out_dim: int = 9) -> Policy:
    """Two-layer MLP with a ZERO output layer (initial policy emits 0 ⇒ default weights)."""
    k1, _k2 = jax.random.split(rng)
    return {
        "W1": jax.random.normal(k1, (obs_dim, hidden)) * 0.5,
        "b1": jnp.zeros(hidden),
        "W2": jnp.zeros((hidden, out_dim)),
        "b2": jnp.zeros(out_dim),
    }


def policy_apply(theta: Policy, obs: jax.Array) -> jax.Array:
    """Return raw per-weight log-multipliers (shape (out_dim,))."""
    h = jnp.tanh(obs @ theta["W1"] + theta["b1"])
    return h @ theta["W2"] + theta["b2"]


def make_theta_to_weights(default_q, default_r,
                          qk: str = "weights_penalization_reference_state_trajectory",
                          rk: str = "weights_penalization_control_squared"):
    """Factory: returns theta_to_weights(theta, obs) -> {qk: Q, rk: R}.

    Q = default_q * exp(log_w[:nq]),  R = default_r * exp(log_w[nq:nq+nr]).
    Zero-init policy ⇒ log_w = 0 ⇒ exp(0)=1 ⇒ Q=default_q, R=default_r (one weight VECTOR
    per env step, broadcast across the MPC horizon — paper-style, not per-stage). Positivity
    via exp keeps the MPC cost convex. NN never outputs a slack weight (no slack in scope).
    """
    default_q = jnp.asarray(default_q)
    default_r = jnp.asarray(default_r)
    nq = int(default_q.shape[0])
    nr = int(default_r.shape[0])

    def theta_to_weights(theta: Policy, obs: jax.Array) -> Dict[str, jax.Array]:
        log_w = policy_apply(theta, obs)  # (nq+nr,)
        q = default_q * jnp.exp(log_w[:nq])
        r = default_r * jnp.exp(log_w[nq:nq + nr])
        return {qk: q, rk: r}

    return theta_to_weights
