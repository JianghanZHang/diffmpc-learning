"""Gates (a)+(b) for the fused-CUDA logbarrier training layer on the QUADROTOR.

Gate (a): fused/jitted forward vs eager filter forward on a training-style stream of
warm-started solves (identical warm starts per step, closed loop advanced by the
FUSED controls), at sqp_tol 1e-3 (training) and 1e-6 (tight).
Gate (b): AD-vs-FD eps ladder (1e-3, 3e-4, 1e-4; plateau check) through the LAYER's
fused path at a warm-started point, sqp_tol=1e-6: dL/dweights (subset) and dL/dx0
(position entries).
"""
import os
import sys
import time

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
import barrier_modes
from barrier_modes import make_barrier_ws_layer, _shift_primal
from turbompc.solvers.backward.logbarrier_backward import (
    logbarrier_nlp_solve, logbarrier_nlp_solve_jit)

QK, RK = env.QK, env.RK


def build():
    dyn, pp = env.build_problem_params()
    layer = make_hard_layer(dyn, pp)
    return dyn, pp, layer.solver


def gate_a(n_steps=12, use_slack=True):
    dyn, pp, solver = build()
    cfgs = barrier_modes.DEFAULT_LB_CFG
    x = jnp.asarray(env.START, dtype=jnp.float64) if hasattr(env, "START") else None
    if x is None:
        x = env.sample_x0(jax.random.PRNGKey(0), 0.0)
    w = {QK: pp[QK], RK: pp[RK]}

    for sqp_tol in (1e-3, 1e-6):
        print(f"\n===== GATE (a)  sqp_tol={sqp_tol:.0e} =====")
        jit_solve = jax.jit(lambda p, ww, s0, c0, _t=sqp_tol: logbarrier_nlp_solve_jit(
            solver, p, ww, s0, c0,
            slack_weight=cfgs["slack_weight"], target_kappa=cfgs["target_kappa"],
            use_slack=use_slack, max_sqp_iter=60, sqp_tol=_t,
            inner_cfg=cfgs["inner_cfg"]))

        # Cold eager solve at x0 (the layer's cold path), then a warm stream.
        pp0 = {**pp, "initial_state": x}
        t0 = time.time()
        res = logbarrier_nlp_solve(
            solver, pp0, w, slack_weight=cfgs["slack_weight"],
            target_kappa=cfgs["target_kappa"], use_slack=use_slack,
            max_sqp_iter=60, sqp_tol=sqp_tol, globalization="filter",
            inner_cfg=cfgs["inner_cfg"])
        print(f"cold eager: it={res['num_iter']} conv={res['final_conv']:.3e} "
              f"t={time.time()-t0:.1f}s")
        guess = _shift_primal(res["states"], res["controls"])
        xk = x
        rows = []
        _orig_ig = solver.program.initial_guess
        try:
            for k in range(n_steps):
                ppk = {**pp, "initial_state": xk}
                s0, c0 = guess
                # eager filter from the SAME warm start
                solver.program.initial_guess = lambda params=None, _g=(s0, c0): _g
                t0 = time.time()
                res_e = logbarrier_nlp_solve(
                    solver, ppk, w, slack_weight=cfgs["slack_weight"],
                    target_kappa=cfgs["target_kappa"], use_slack=use_slack,
                    max_sqp_iter=60, sqp_tol=sqp_tol, globalization="filter",
                    inner_cfg=cfgs["inner_cfg"])
                t_e = time.time() - t0
                solver.program.initial_guess = _orig_ig
                # fused jit from the SAME warm start
                t0 = time.time()
                sol_j = jax.block_until_ready(jit_solve(ppk, w, s0, c0))
                t_j = time.time() - t0
                rl_s = float(jnp.max(jnp.abs(sol_j.states - res_e["states"])) /
                             (jnp.max(jnp.abs(res_e["states"])) + 1e-30))
                rl_c = float(jnp.max(jnp.abs(sol_j.controls - res_e["controls"])) /
                             (jnp.max(jnp.abs(res_e["controls"])) + 1e-30))
                rows.append((k, res_e["num_iter"], int(sol_j.num_iter),
                             res_e["final_conv"], float(sol_j.final_conv),
                             rl_s, rl_c, t_e, t_j))
                print(f"  step {k:2d}: eager it={res_e['num_iter']:2d} "
                      f"conv={res_e['final_conv']:.2e} t={t_e:5.2f}s | "
                      f"jit it={int(sol_j.num_iter):2d} conv={float(sol_j.final_conv):.2e} "
                      f"t={t_j:5.2f}s | rel(s)={rl_s:.2e} rel(c)={rl_c:.2e}")
                # advance closed loop with the FUSED first control
                u0 = jax.lax.stop_gradient(sol_j.controls[0])
                xk = jax.lax.stop_gradient(env.simulate_step(dyn, xk, u0))
                guess = _shift_primal(sol_j.states, sol_j.controls)
        finally:
            solver.program.initial_guess = _orig_ig
        r = np.array([[q[5], q[6]] for q in rows])
        tj = np.array([q[8] for q in rows])
        print(f"  SUMMARY tol={sqp_tol:.0e}: max rel(s)={r[:,0].max():.2e} "
              f"max rel(c)={r[:,1].max():.2e}  jit conv all "
              f"<{max(q[4] for q in rows):.2e}  warm jit t: first={tj[0]:.2f}s "
              f"steady={np.median(tj[1:]):.3f}s")


def gate_b(use_slack=True):
    dyn, pp, solver = build()
    x = env.sample_x0(jax.random.PRNGKey(0), 0.0)
    w0 = {QK: pp[QK], RK: pp[RK]}

    blayer = make_barrier_ws_layer(solver, cfg={"sqp_tol": 1e-6, "max_sqp_iter": 60,
                                                "forward": "fused", "use_slack": use_slack})
    ppx = {**pp, "initial_state": x}
    # Warm the layer: cold eager solve + a few fused solves at the same x (warm cell).
    for _ in range(3):
        blayer.solve(ppx, w0)
    snap = blayer.snapshot()

    def loss_w(wq, wr):
        blayer.restore(snap)
        st, co = blayer.solve(ppx, {QK: wq, RK: wr})
        return env.task_loss(st, co)

    print("\n===== GATE (b)  AD vs FD (eps ladder, plateau check) =====")
    g_q, g_r = jax.grad(loss_w, argnums=(0, 1))(pp[QK], pp[RK])
    g_ad = np.concatenate([np.asarray(g_q).ravel(), np.asarray(g_r).ravel()])

    # FD over ALL weight entries, eps ladder with plateau check per entry
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
    # plateau: last two ladder steps agree
    plateau_rel = np.abs(fd[-1] - fd[-2]) / (np.abs(fd[-1]) + 1e-12)
    flagged = plateau_rel > 2e-3
    g_fd = fd[-1]
    ok = ~flagged
    cos = float(g_ad[ok] @ g_fd[ok] /
                (np.linalg.norm(g_ad[ok]) * np.linalg.norm(g_fd[ok]) + 1e-30))
    rel = float(np.linalg.norm(g_ad[ok] - g_fd[ok]) /
                (np.linalg.norm(g_fd[ok]) + 1e-30))
    print(f"weights: dim={dim} flagged={int(flagged.sum())} cos={cos:.6f} rel={rel:.2e}")
    print(f"  AD  = {np.array2string(g_ad, precision=3)}")
    print(f"  FD  = {np.array2string(g_fd, precision=3)}")

    # dL/dx0 over the 3 position entries
    def loss_x(x0):
        blayer.restore(snap)
        st, co = blayer.solve({**pp, "initial_state": x0}, w0)
        return env.task_loss(st, co)

    gx_ad = np.asarray(jax.grad(loss_x)(x))[:3]
    fdx = np.zeros((len(eps_seq), 3))
    for i in range(3):
        for je, eps in enumerate(eps_seq):
            def _fx(delta, _i=i):
                xp = np.array(x, dtype=float)
                xp[_i] += delta
                return float(loss_x(jnp.asarray(xp)))
            fdx[je, i] = (_fx(+eps) - _fx(-eps)) / (2 * eps)
    plateau_x = np.abs(fdx[-1] - fdx[-2]) / (np.abs(fdx[-1]) + 1e-12)
    flagged_x = plateau_x > 2e-3
    okx = ~flagged_x
    cosx = float(gx_ad[okx] @ fdx[-1][okx] /
                 (np.linalg.norm(gx_ad[okx]) * np.linalg.norm(fdx[-1][okx]) + 1e-30))
    relx = float(np.linalg.norm(gx_ad[okx] - fdx[-1][okx]) /
                 (np.linalg.norm(fdx[-1][okx]) + 1e-30))
    print(f"x0[:3]: flagged={int(flagged_x.sum())} cos={cosx:.6f} rel={relx:.2e}")
    print(f"  AD = {gx_ad}  FD = {fdx[-1]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    use_slack = not (len(sys.argv) > 2 and sys.argv[2] == "pure")
    print(f"mode = {'elastic' if use_slack else 'PURE barrier'}")
    if "a" in which:
        gate_a(use_slack=use_slack)
    if "b" in which:
        gate_b(use_slack=use_slack)
