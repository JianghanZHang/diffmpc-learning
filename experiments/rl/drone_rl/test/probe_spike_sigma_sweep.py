"""Deterministic replay of the OFF update-61 spike (grad 100.9): reconstruct the
V4 window gradient at that exact (theta, x_before, warm cell) and sweep sigma_x /
sigma_f to measure how each §4.7 regularizer damps the spike.
"""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_TURBOMPC = os.path.normpath(os.path.join(_PKG, "../../../external/diffmpc2"))
for _p in (_PKG, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from env import quadrotor_env as env
from mpc_layer import make_hard_layer
from policy import make_theta_to_weights
from barrier_modes import make_barrier_ws_layer

QK, RK, H = env.QK, env.RK, 24
dyn, pp = env.build_problem_params()
solver = make_hard_layer(dyn, pp).solver
t2w = make_theta_to_weights(pp[QK], pp[RK])

d = np.load(os.path.join(_PKG, "trained_policies", "spike_quadrotor_bptt_barrier_pure_upd61.npz"))
theta = {k: jnp.asarray(d[k]) for k in ("W1", "W2", "b1", "b2")}
x0 = jnp.asarray(d["x_before"])
guess = (jnp.asarray(d["guess_states"]), jnp.asarray(d["guess_controls"]))


def grad_norm(sigma_x, sigma_f):
    lay = make_barrier_ws_layer(solver, cfg={
        "use_slack": False, "bwd_sigma_x": sigma_x, "bwd_sigma_f": sigma_f})

    def window_loss(phi):
        lay.restore(guess)                 # reset warm cell to the spike-update start
        xw = x0
        L = jnp.zeros((), dtype=xw.dtype)
        for _t in range(H):
            w = t2w(phi, xw)
            st, co = lay.solve({**pp, "initial_state": xw}, w)
            xw = env.simulate_step(dyn, xw, co[0])
            L = L + env.task_loss(xw[None], co[0][None])
        return L

    g = jax.grad(window_loss)(theta)
    return float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in jax.tree.leaves(g))))


print("=== spike upd-61 replay: ||grad|| vs regularizer (OFF=100.9 in training) ===")
print("sigma_x sweep (sigma_f=0):")
for sx in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
    print(f"  sigma_x={sx:<6}: ||grad||={grad_norm(sx, 0.0):.3f}")
print("sigma_f sweep (sigma_x=0):")
for sf in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
    print(f"  sigma_f={sf:<6}: ||grad||={grad_norm(0.0, sf):.3f}")
print("combined:")
for sx, sf in [(1e-2, 1e-2), (1e-1, 1e-1)]:
    print(f"  sigma_x={sx} sigma_f={sf}: ||grad||={grad_norm(sx, sf):.3f}")
