"""Where does V4p's gradient blowup come from? Three direct measurements.

(a) W at a PINNED state: pure W = y^2/kappa (uncapped) vs elastic W = y/(s+y/gamma)
    (capped at gamma) — evaluated at each mode's own fixed point, same x, same theta.
(b) Per-step feedback gain ||dU0/dx0||_2 through each layer at the same pinned state.
(c) Window-gradient norm vs BPTT length h (V4-style live window, same theta = the
    trained V4p policy, same start state) for pure vs elastic layers.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                  # drone_rl/
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
from turbompc.solvers.admm.logbarrier_admm_qp import to_one_sided
from turbompc.solvers.admm.admm import _apply_G
from turbompc.solvers.qp_utils import pack_x

KAPPA, GAMMA = 1e-4, 1e2
QK, RK = env.QK, env.RK

dyn, pp = env.build_problem_params()
solver = make_hard_layer(dyn, pp).solver
t2w = make_theta_to_weights(pp[QK], pp[RK])

d = np.load(f"{_PKG}/trained_policies/train_quadrotor_bptt_barrier_pure_seed0_theta.npz")
theta = {k: jnp.asarray(d[k]) for k in d.files}

lay_p = make_barrier_ws_layer(solver, cfg={"use_slack": False})
lay_e = make_barrier_ws_layer(solver, cfg={"use_slack": True})

# ---- drive the PURE closed loop from START into the grazing phase ----------
x = jnp.array(env.START, dtype=jnp.float64)
margins = []
X_TRAJ = [x]
for t in range(14):
    w = t2w(theta, x)
    st, co = lay_p.solve({**pp, "initial_state": x}, w)
    x = jax.lax.stop_gradient(env.simulate_step(dyn, x, co[0]))
    X_TRAJ.append(x)
    margins.append(float(env.obs_margin(x)))
margins = np.array(margins)
k_pin = int(np.argmax(margins))          # most-pinned realized state
x_pin = X_TRAJ[k_pin + 1]
print(f"closed-loop margins (steps 1-14): {np.array2string(margins, precision=4)}")
print(f"pinned state: step {k_pin+1}, margin {margins[k_pin]:+.5f}")

# ---- (a) W at the pinned state, each mode at its own fixed point ------------
def w_stats(lay, use_slack, tag):
    w = t2w(theta, x_pin)
    st, co = lay.solve({**pp, "initial_state": x_pin}, w)
    pp_w = solver.make_params_with_weights(w, {**pp, "initial_state": x_pin})
    qp = solver._build_qp_data(st, co, pp_w)
    qp1 = to_one_sided(qp, GAMMA, use_slack=use_slack)
    delta = _apply_G(qp1, pack_x(st, co)) - qp1.ineq.u          # G1 x - h
    mask = jnp.abs(qp1.ineq.u) < 1e7                            # real rows only
    if use_slack:
        disc = jnp.sqrt(delta * delta + 4.0 * KAPPA / GAMMA)
        s = (-delta + disc) / 2.0
        y = KAPPA / s
        W = y / (s + y / GAMMA)
    else:
        s = jnp.maximum(-delta, 1e-30)
        y = KAPPA / s
        W = y / s
    W = jnp.where(mask, W, 0.0)
    print(f"[{tag}] min s(real rows) = {float(jnp.min(jnp.where(mask, s, jnp.inf))):.3e}   "
          f"max y = {float(jnp.max(jnp.where(mask, y, 0.0))):.3e}   "
          f"max W = {float(jnp.max(W)):.3e}")

# warm the ELASTIC cell from the pure layer's primal snapshot (cells are
# mode-agnostic (states, controls) guesses) — avoids a ~60 s cold eager solve.
lay_e.reset(lay_p.snapshot())
w_stats(lay_p, False, "pure   ")
w_stats(lay_e, True,  "elastic")

# ---- (b) per-step feedback gain ||dU0/dx0|| ---------------------------------
def feedback_gain(lay, tag):
    w = t2w(theta, x_pin)
    lay.solve({**pp, "initial_state": x_pin}, w)   # cell already warm at x_pin
    snap = lay.snapshot()

    def u0_of_x(x0):
        lay.restore(snap)
        st, co = lay.solve({**pp, "initial_state": x0}, w)
        return co[0]

    K = jax.jacrev(u0_of_x)(x_pin)                 # (nu, nx)
    sv = np.linalg.svd(np.asarray(K), compute_uv=False)
    print(f"[{tag}] ||dU0/dx0||_2 = {sv[0]:.3e}   top-3 sv = {np.array2string(sv[:3], precision=3)}")
    return sv[0]

k_p = feedback_gain(lay_p, "pure   ")
k_e = feedback_gain(lay_e, "elastic")

# ---- (c) window-grad norm vs h (V4-style live window, same theta/start) -----
def window_grad_norm(lay, h, x_start, snap):
    def loss(phi):
        lay.restore(snap)
        xw = x_start
        L = jnp.zeros((), dtype=xw.dtype)
        mx = -jnp.inf
        for _t in range(h):
            w = t2w(phi, xw)
            st, co = lay.solve({**pp, "initial_state": xw}, w)
            u0 = co[0]
            xw = env.simulate_step(dyn, xw, u0)
            L = L + env.task_loss(xw[None], u0[None])
            mx = jnp.maximum(mx, env.obs_margin(xw))
        return L, mx

    (L, mx), g = jax.value_and_grad(loss, has_aux=True)(theta)
    gn = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in jax.tree.leaves(g))))
    return gn, float(mx)

x_start = X_TRAJ[max(k_pin - 2, 0)]     # start the window just before the pinned phase
print(f"\nwindow start: step {max(k_pin-2,0)}, margin "
      f"{float(env.obs_margin(x_start)):+.5f}")
snaps = {}
for lay, tag in ((lay_p, "p"), (lay_e, "e")):
    lay.solve({**pp, "initial_state": x_start}, t2w(theta, x_start))  # warm at x_start
    snaps[tag] = lay.snapshot()
print(f"{'h':>4} {'|g| pure':>12} {'|g| elastic':>12} {'maxmargin p':>12} {'maxmargin e':>12}", flush=True)
for h in (1, 2, 4, 8, 16, 24):
    gp, mp = window_grad_norm(lay_p, h, x_start, snaps["p"])
    ge, me = window_grad_norm(lay_e, h, x_start, snaps["e"])
    print(f"{h:>4} {gp:>12.4e} {ge:>12.4e} {mp:>+12.5f} {me:>+12.5f}", flush=True)
