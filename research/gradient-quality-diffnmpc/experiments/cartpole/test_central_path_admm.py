import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from central_path_admm import retraction_map, elastic_retraction  # noqa: E402


def test_retraction_map_complementarity_and_relu_limit():
    v = jnp.asarray(np.linspace(-5, 5, 21))
    g = 0.3
    assert float(jnp.max(jnp.abs(retraction_map(v, g) * retraction_map(-v, g) - g))) < 1e-10
    assert float(jnp.max(jnp.abs(retraction_map(v, 1e-9) - jnp.maximum(v, 0.0)))) < 1e-4


def test_elastic_retraction_inactive_row_recovers_target_as_kappa_to_zero():
    h = jnp.array([2.0])
    z_tilde = jnp.array([0.5])          # below the bound -> inactive
    z_g, xi = elastic_retraction(z_tilde, h, kappa=1e-8, rho=0.3, slack_weight=1.0)
    assert float(jnp.abs(z_g[0] - 0.5)) < 1e-4 and float(xi[0]) < 1e-4


def test_elastic_retraction_active_row_matches_soft_blend_and_slack_engages():
    h = jnp.array([2.0])
    z_tilde = jnp.array([7.0])          # above the bound -> active/violated
    rho, gamma, kappa = 0.3, 0.1, 1e-6
    z_g, xi = elastic_retraction(z_tilde, h, kappa, rho, gamma)
    soft = h[0] + (z_tilde[0] - h[0]) * rho / (gamma + rho)   # frac-blend / soft solution
    assert float(jnp.abs(z_g[0] - soft)) < 1e-3
    assert float(xi[0]) > 0.0 and float(z_g[0]) > float(h[0])  # slack engages, constraint relaxed
