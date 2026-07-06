"""TDD tests for policy.py and optimizer.py (drone Diff-WMPC pipeline)."""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # drone_rl/

import jax.numpy as jnp
import pytest

from policy import init_policy, policy_apply, make_theta_to_weights
from optimizer import adam_init, adam_step

QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"


def test_zero_init_gives_defaults():
    """Zero-init output layer => log_w=0 => exp(0)=1 => outputs equal default weights."""
    theta = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=9)
    default_q = jnp.array([0.1] * 6)
    default_r = jnp.array([1.0] * 3)
    t2w = make_theta_to_weights(default_q, default_r, qk=QK, rk=RK)

    for obs in [
        jnp.zeros(6),
        jnp.ones(6),
        jax.random.normal(jax.random.PRNGKey(42), (6,)),
    ]:
        weights = t2w(theta, obs)
        assert jnp.allclose(weights[QK], default_q, atol=1e-12), (
            f"Q mismatch: {weights[QK]} vs {default_q}"
        )
        assert jnp.allclose(weights[RK], default_r, atol=1e-12), (
            f"R mismatch: {weights[RK]} vs {default_r}"
        )


def test_policy_apply_differentiable():
    """policy_apply must be JAX-differentiable; all gradient leaves must be finite."""
    theta = init_policy(jax.random.PRNGKey(1), obs_dim=6, out_dim=9)
    # Perturb W2 so that the gradient through the output layer is non-trivial.
    theta = {**theta, "W2": jax.random.normal(jax.random.PRNGKey(7), theta["W2"].shape) * 0.1}
    obs = jax.random.normal(jax.random.PRNGKey(2), (6,))

    grads = jax.grad(lambda th: jnp.sum(policy_apply(th, obs)))(theta)

    for key, g in grads.items():
        assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient leaf: {key}"


def test_theta_to_weights_positive_and_shapes():
    """With non-zero W2, outputs must be positive and have the correct shapes."""
    theta = init_policy(jax.random.PRNGKey(3), obs_dim=6, out_dim=9)
    theta = {**theta, "W2": jax.random.normal(jax.random.PRNGKey(8), theta["W2"].shape) * 0.5}

    default_q = jnp.array([0.1] * 6)
    default_r = jnp.array([1.0] * 3)
    t2w = make_theta_to_weights(default_q, default_r, qk=QK, rk=RK)
    obs = jax.random.normal(jax.random.PRNGKey(4), (6,))

    weights = t2w(theta, obs)
    assert weights[QK].shape == (6,), f"Q shape wrong: {weights[QK].shape}"
    assert weights[RK].shape == (3,), f"R shape wrong: {weights[RK].shape}"
    assert jnp.all(weights[QK] > 0), f"Q not positive: {weights[QK]}"
    assert jnp.all(weights[RK] > 0), f"R not positive: {weights[RK]}"


def test_adam_reduces_quadratic():
    """Adam must drive a simple quadratic loss to near-zero in ~200 steps."""
    params = {"w": jnp.zeros(4)}
    target = 3.0

    def loss_fn(p):
        return jnp.sum((p["w"] - target) ** 2)

    state = adam_init(params)
    for _ in range(200):
        grads = jax.grad(loss_fn)(params)
        params, state = adam_step(params, grads, state, lr=0.1)

    final_loss = loss_fn(params)
    assert final_loss < 1e-3, f"Adam did not converge: final loss = {final_loss}"
