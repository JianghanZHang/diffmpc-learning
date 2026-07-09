"""Profile the exact Moré–Sorensen trust-region adjoint on a REAL quadrotor backward
(structured dynamics Jacobians — well-conditioned saddle), not the synthetic random-C
instance that made cuDSS non-deterministic. Also confirms cuDSS repeatability is ~1e-5
(not 9%) on the real system.
"""
import sys, time
import os
_HERE=os.path.dirname(os.path.abspath(__file__))
_PKG=os.path.dirname(_HERE)
_TURBOMPC=os.path.normpath(os.path.join(_PKG,"../../../external/diffmpc2"))
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
import barrier_modes
from turbompc.solvers.backward import logbarrier_backward as L

QK, RK = env.QK, env.RK
dyn, pp = env.build_problem_params()
solver = make_hard_layer(dyn, pp).solver
t2w = make_theta_to_weights(pp[QK], pp[RK])
cfg = barrier_modes.DEFAULT_LB_CFG

# a warm-started converged pure-barrier solve (real, well-conditioned backward)
x0 = jnp.array(env.START, dtype=jnp.float64)
w = {QK: pp[QK], RK: pp[RK]}
lay = barrier_modes.make_barrier_ws_layer(solver, cfg={"use_slack": False, "sqp_tol": 1e-6})
ppx = {**pp, "initial_state": x0}
for _ in range(3):
    lay.solve(ppx, w)
# grab the forward duals for the backward
from turbompc.solvers.backward.logbarrier_backward import logbarrier_nlp_solve_jit
sol = logbarrier_nlp_solve_jit(solver, ppx, w, *lay.snapshot(),
                               slack_weight=cfg["slack_weight"], target_kappa=cfg["target_kappa"],
                               use_slack=False, max_sqp_iter=60, sqp_tol=1e-6,
                               inner_cfg=cfg["inner_cfg"], lifted=True)
states_c, controls_c = sol.states, sol.controls
rng = np.random.default_rng(1)
d_states = jnp.asarray(rng.standard_normal(np.asarray(states_c).shape))
d_controls = jnp.asarray(rng.standard_normal(np.asarray(controls_c).shape))

kw = dict(slack_weight=cfg["slack_weight"], use_slack=False, include_ineq_hessian=True,
          y_f_dyn_c=sol.y_f_dyn, y_g_stacked_c=sol.y_g_stacked, yg_mode="cached",
          yg_crosscheck_tol=None, pure_gamma_smooth=1e8)

def bwd(**extra):
    gw, gx, _ = L._relaxed_nlp_backward(
        solver, ppx, w, states_c, controls_c, 1e-4, d_states, d_controls, **kw, **extra)
    return gw, gx

# plain gradient + its dL/dx0 norm (the quantity a trust region would cap)
gw0, gx0 = bwd()
def gnorm(gw, gx):
    return float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gw)) + jnp.sum(gx**2)))
n0 = gnorm(gw0, gx0)
print(f"real quadrotor backward: plain ||grad|| = {n0:.4f}")

# cuDSS repeatability on the REAL backward (the 'floor')
r1 = gnorm(*bwd()); r2 = gnorm(*bwd())
print(f"cuDSS repeatability (real backward): ||g|| = {n0:.6f} / {r1:.6f} / {r2:.6f} "
      f"(spread {abs(r1-r2)/n0:.2e})")

def timeit(fn, reps=5):
    fn(); t=time.time()
    for _ in range(reps): jax.block_until_ready(fn())
    return (time.time()-t)/reps

t_plain = timeit(lambda: bwd())
# trust region: cap at half the plain adjoint norm (forces activation)
lam_x0, _ = L.solve_reduced_relaxed_kkt_cudss(  # get the ADJOINT norm for Delta
    L._noop if False else None, None, None, None) if False else (None, None)
# use dL/dx0 as the adjoint-magnitude proxy target isn't exposed; use gradient norm cap via Delta on lam_x:
# run trust with a few Delta values relative to the internal lam_x norm
for frac in (2.0, 0.5, 0.2):
    # Delta relative to... we need the lam_x norm; approximate by scaling: run once to read info
    gw, gx = bwd(trust_radius=1e9)   # huge Delta -> interior, reads lam_x norm
    lamnrm = L.LAST_TRUST_INFO_HOLDER[0]["nrm"]
    Delta = frac * lamnrm
    gw, gx = bwd(trust_radius=Delta)
    info = L.LAST_TRUST_INFO_HOLDER[0]
    t_tr = timeit(lambda: bwd(trust_radius=Delta))
    print(f"  Delta={frac}x||lam||: sigma*={info['sigma']:.3e} newton={info['n_newton']} "
          f"solves={info['n_solve']} ||lam||/Delta={info['nrm']/Delta:.4f} interior={info['interior']} "
          f"| plain={t_plain*1e3:.1f}ms trust={t_tr*1e3:.1f}ms ratio={t_tr/t_plain:.1f}x")
