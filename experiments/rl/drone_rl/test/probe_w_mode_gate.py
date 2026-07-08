"""Gates for the dual-only W fold (pure backward): W = y^2/kappa from cached duals.

Gate 1 (training tol, the fix's raison d'etre): at sqp_tol=1e-3, drive the pure
closed loop to the pinned phase, find the state where the raw clearance crosses
(clip fires), and compare the point/window gradient of OLD (analytic + clearance-W)
vs NEW (cached + dual-W) against a TIGHT reference (sqp_tol=1e-6, analytic — the
config that passed gate (b) pure).

Gate 2 (correctness at tight tol): AD-vs-FD eps ladder for the NEW config at
sqp_tol=1e-6 (17 weight entries + x0[:3]), same protocol as fused_gates gate (b).
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


def min_raw_s(st, co, w, x0):
    """min over REAL one-sided rows of the raw clearance h - G1 x at the plan."""
    pp_w = solver.make_params_with_weights(w, {**pp, "initial_state": x0})
    qp = solver._build_qp_data(st, co, pp_w)
    qp1 = to_one_sided(qp, GAMMA, use_slack=False)
    delta = _apply_G(qp1, pack_x(st, co)) - qp1.ineq.u
    mask = jnp.abs(qp1.ineq.u) < 1e7
    return float(jnp.min(jnp.where(mask, -delta, jnp.inf)))


def grad_stats(g):
    v = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(g)])
    return v


def cos_rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
    return cos, rel


# --------------------------------------------------------------------------- #
# Layers: OLD/NEW at training tol (forwards identical), REF at tight tol.
# --------------------------------------------------------------------------- #
lay_old = make_barrier_ws_layer(solver, cfg={
    "use_slack": False, "bwd_w_cap": None, "bwd_pure_smooth_gamma": None})   # UNCAPPED (old)
lay_new = make_barrier_ws_layer(solver, cfg={"use_slack": False})            # SMOOTHED g_eff=1e8 (new default)
lay_clamp = make_barrier_ws_layer(solver, cfg={
    "use_slack": False, "bwd_w_cap": 1e8, "bwd_pure_smooth_gamma": None})    # hard clamp
# CASE 1 (user): y = kappa/s_raw reconstructed at states_c (NO smoothing), W = y^2/kappa
lay_mani = make_barrier_ws_layer(solver, cfg={
    "use_slack": False, "bwd_w_from_dual": True, "bwd_pure_smooth_gamma": None})
# CASE 2 (user): y CACHED from the forward pass (retraction dual, one anchor back)
lay_c2 = make_barrier_ws_layer(solver, cfg={
    "use_slack": False, "bwd_yg_mode": "cached", "bwd_w_from_dual": True,
    "bwd_pure_smooth_gamma": None})
lay_ref = make_barrier_ws_layer(solver, cfg={
    "use_slack": False, "sqp_tol": 1e-6, "max_sqp_iter": 60,
    "bwd_w_cap": None, "bwd_pure_smooth_gamma": None})

# ---- drive the pure closed loop; log min raw clearance per solve ------------
x = jnp.array(env.START, dtype=jnp.float64)
traj, s_mins = [x], []
for t in range(14):
    w = t2w(theta, x)
    st, co = lay_new.solve({**pp, "initial_state": x}, w)
    s_mins.append(min_raw_s(st, co, w, x))
    x = jax.lax.stop_gradient(env.simulate_step(dyn, x, co[0]))
    traj.append(x)
s_mins = np.array(s_mins)
print("min raw clearance per solve along the drive:")
print(" ", np.array2string(s_mins, precision=2))
k_cross = int(np.argmin(s_mins))
x_star = traj[k_cross]                      # solve AT this state had min raw s
print(f"worst solve: step {k_cross}, min raw s = {s_mins[k_cross]:.3e} "
      f"({'CROSSED/clip' if s_mins[k_cross] <= 1e-12 else 'positive'})")

# share one warm snapshot across all layers at x_star
snap = lay_new.snapshot() if k_cross == len(s_mins) - 1 else None
_ALL = (lay_old, lay_new, lay_clamp, lay_mani, lay_c2, lay_ref)
for lay in _ALL:
    lay.reset(lay_new.snapshot())
w_star = t2w(theta, x_star)
for lay in _ALL:
    lay.solve({**pp, "initial_state": x_star}, w_star)   # warm all cells at x_star
snaps = {id(l): l.snapshot() for l in _ALL}

# --------------------------------------------------------------------------- #
# Gate 1: point + window gradients at the crossing state, old vs new vs ref
# --------------------------------------------------------------------------- #
def point_grad(lay):
    def loss(phi):
        lay.restore(snaps[id(lay)])
        st, co = lay.solve({**pp, "initial_state": x_star}, t2w(phi, x_star))
        return env.task_loss(st, co)
    return jax.grad(loss)(theta)

def window_grad(lay, h=4):
    def loss(phi):
        lay.restore(snaps[id(lay)])
        xw = x_star
        L = jnp.zeros((), dtype=xw.dtype)
        for _t in range(h):
            w = t2w(phi, xw)
            st, co = lay.solve({**pp, "initial_state": xw}, w)
            xw = env.simulate_step(dyn, xw, co[0])
            L = L + env.task_loss(xw[None], co[0][None])
        return L
    return jax.grad(loss)(theta)

print("\n===== GATE 1: uncapped / clamp / SMOOTH at the crossing state (vs 1e-6 ref) =====")
for tag, fn in (("point", point_grad), ("window h=4", window_grad)):
    g_ref = grad_stats(fn(lay_ref))
    for nm, lay in (("uncapped", lay_old), ("clamp1e8", lay_clamp), ("smooth1e8", lay_new),
                    ("case1:k/s_raw", lay_mani), ("case2:y_fwd", lay_c2)):
        g = grad_stats(fn(lay))
        c, r = cos_rel(g, g_ref)
        print(f"[{tag:10s}] {nm:9s} vs ref: cos={c:+.6f} rel={r:.3e}  |g|={np.linalg.norm(g):.3e}")

# --------------------------------------------------------------------------- #
# Gate 2: AD-vs-FD ladder for the NEW config at tight tol (protocol of gate (b))
# --------------------------------------------------------------------------- #
print("\n===== GATE 2: AD vs FD, SMOOTHED config, sqp_tol=1e-6 =====")
lay_g = make_barrier_ws_layer(solver, cfg={"use_slack": False, "sqp_tol": 1e-6,
                                           "max_sqp_iter": 60})     # capped (new default)
x0g = env.sample_x0(jax.random.PRNGKey(0), 0.0)
lay_g.reset(lay_new.snapshot())
ppx = {**pp, "initial_state": x0g}
w0 = {QK: pp[QK], RK: pp[RK]}
for _ in range(3):
    lay_g.solve(ppx, w0)
snapg = lay_g.snapshot()

def loss_w(wq, wr):
    lay_g.restore(snapg)
    st, co = lay_g.solve(ppx, {QK: wq, RK: wr})
    return env.task_loss(st, co)

g_q, g_r = jax.grad(loss_w, argnums=(0, 1))(pp[QK], pp[RK])
g_ad = np.concatenate([np.asarray(g_q).ravel(), np.asarray(g_r).ravel()])

eps_seq = (1e-3, 3e-4, 1e-4)
nq = np.asarray(pp[QK]).size
dim = nq + np.asarray(pp[RK]).size
fd = np.zeros((len(eps_seq), dim))
for i in range(dim):
    for je, eps in enumerate(eps_seq):
        def _f(delta, _i=i):
            wq = np.array(pp[QK], dtype=float)
            wr = np.array(pp[RK], dtype=float)
            if _i < nq:
                wq[_i] += delta
            else:
                wr[_i - nq] += delta
            return float(loss_w(jnp.asarray(wq), jnp.asarray(wr)))
        fd[je, i] = (_f(+eps) - _f(-eps)) / (2 * eps)
plateau_rel = np.abs(fd[-1] - fd[-2]) / (np.abs(fd[-1]) + 1e-12)
flagged = plateau_rel > 2e-3
ok = ~flagged
cos, rel = cos_rel(g_ad[ok], fd[-1][ok])
print(f"weights: dim={dim} flagged={int(flagged.sum())} cos={cos:.6f} rel={rel:.2e}")

def loss_x(x0):
    lay_g.restore(snapg)
    st, co = lay_g.solve({**pp, "initial_state": x0}, w0)
    return env.task_loss(st, co)

gx_ad = np.asarray(jax.grad(loss_x)(x0g))[:3]
fdx = np.zeros((len(eps_seq), 3))
for i in range(3):
    for je, eps in enumerate(eps_seq):
        def _fx(delta, _i=i):
            xp = np.array(x0g, dtype=float)
            xp[_i] += delta
            return float(loss_x(jnp.asarray(xp)))
        fdx[je, i] = (_fx(+eps) - _fx(-eps)) / (2 * eps)
flag_x = (np.abs(fdx[-1] - fdx[-2]) / (np.abs(fdx[-1]) + 1e-12)) > 2e-3
okx = ~flag_x
cosx, relx = cos_rel(gx_ad[okx], fdx[-1][okx])
print(f"x0[:3]: flagged={int(flag_x.sum())} cos={cosx:.6f} rel={relx:.2e}")
