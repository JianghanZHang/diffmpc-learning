"""Verification gate for the NONLINEAR quadrotor Diff-WMPC env (run before training).

Stages (cuDSS env):
  [0] API parity: quadrotor_env exposes the same public names as drone_env.
  [1] Forward solve from START (zero-init policy -> default weights): convergence,
      SQP iters, and whether the obstacle is ENGAGED in the plan.
  [2] Closed-loop rollout (default weights): per-step REALIZED obstacle margin
      (does the realized path GRAZE the boundary?), goal distance, quaternion-norm
      drift, violations.
  [3] V3 BPTT directional finite-difference gate: cos(AD, FD) over the gradient
      direction + mixed directions on a short (h=3) window -> the gradient is correct.

Run::
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/diffmpc2 \\
      /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
      experiments/rl/drone_rl/verify_quadrotor.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "diffmpc2")
for _p in (_HERE, _SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from env import quadrotor_env as env
from env import drone_env
from mpc_layer import make_hard_layer
from policy import init_policy, make_theta_to_weights
from gradient_modes import make_initial_guess, prime_guess, shift_guess, make_shac_window_loss


def stage0_api_parity():
    print("\n[0] API parity (quadrotor_env vs drone_env) ...")
    needed = ["NX", "NU", "START", "GOAL", "QK", "RK",
              "build_problem_params", "simulate_step", "task_loss",
              "obs_margin", "goal_dist", "sample_x0"]
    missing = [n for n in needed if not hasattr(env, n)]
    assert not missing, f"quadrotor_env missing: {missing}"
    drone_missing = [n for n in needed if not hasattr(drone_env, n)]
    assert not drone_missing, f"drone_env missing: {drone_missing}"
    print(f"    OK  NX={env.NX} NU={env.NU}  HORIZON={env.HORIZON} DT={env.DT} "
          f"scheme=RK4  HOVER_THRUST={env.HOVER_THRUST:.4f}  UMAX={env.UMAX}")
    print(f"    OBS_C={np.asarray(env.OBS_C)} OBS_R={env.OBS_R}  START_pos={np.asarray(env.START[:3])}")


def _build():
    dyn, pp = env.build_problem_params()
    layer = make_hard_layer(dyn, pp)
    solver = layer.solver
    t2w = make_theta_to_weights(pp[env.QK], pp[env.RK])
    policy = init_policy(jax.random.PRNGKey(0), obs_dim=env.NX, out_dim=env.NX + env.NU)
    return dyn, pp, solver, t2w, policy


def stage1_forward(dyn, pp, solver, t2w, policy):
    print("\n[1] Forward solve from START (default weights) ...")
    x0 = jnp.asarray(env.START)
    w = t2w(policy, x0)  # zero-init -> defaults
    ig = solver.initial_guess({**pp, "initial_state": x0})
    t = time.time()
    sol = solver.solve(ig, {**pp, "initial_state": x0}, w)
    jax.block_until_ready(sol.states)
    dt = time.time() - t
    conv = float(sol.convergence_error)
    nit = int(sol.num_iter)
    plan_margin = np.asarray(env.obs_margin(sol.states))
    ctrl = np.asarray(sol.controls)
    print(f"    conv_err={conv:.2e}  SQP_iters={nit}  solve={dt:.2f}s (incl compile)")
    print(f"    plan obs_margin: max={plan_margin.max():+.4f} (>~0 ⇒ ENGAGED)  "
          f"min={plan_margin.min():+.4f}  #steps>=-0.05={(plan_margin>=-0.05).sum()}/{plan_margin.size}")
    print(f"    control range: thrust[{ctrl[:,0].min():.3f},{ctrl[:,0].max():.3f}] "
          f"(hover={env.HOVER_THRUST:.3f}, umax={env.UMAX})  "
          f"|tau|max={np.abs(ctrl[:,1:]).max():.3f}")
    print(f"    plan reaches: final pos={np.asarray(sol.states[-1,:3])}  "
          f"goal_dist={float(env.goal_dist(sol.states[-1])):.3f}")
    return sol


def stage2_rollout(dyn, pp, solver, t2w, policy, n_steps=35):
    print(f"\n[2] Closed-loop rollout (default weights, {n_steps} steps) ...")
    x = jax.lax.stop_gradient(jnp.asarray(env.START))
    guess = make_initial_guess(solver, pp, x)
    guess = prime_guess(solver, t2w, policy, pp, x, guess)

    jsolve = jax.jit(lambda g, xx, ww: solver.solve(g, {**pp, "initial_state": xx}, ww))
    margins, dists, qnorms, iters, traj = [], [], [], [], [np.asarray(x[:3])]
    for k in range(n_steps):
        w = t2w(policy, jax.lax.stop_gradient(x))
        sol = jsolve(guess, x, w)
        jax.block_until_ready(sol.states)
        u0 = jax.lax.stop_gradient(sol.controls[0])
        x = jax.lax.stop_gradient(env.simulate_step(dyn, x, u0))
        guess = jax.lax.stop_gradient(shift_guess(sol))
        margins.append(float(env.obs_margin(x)))
        dists.append(float(env.goal_dist(x)))
        qnorms.append(float(jnp.linalg.norm(x[6:10])))
        iters.append(int(sol.num_iter))
        traj.append(np.asarray(x[:3]))
    margins = np.array(margins)
    n_viol = int((margins > 0).sum())
    n_graze = int((margins > -0.1).sum())  # steps within 0.1 of the boundary
    print(f"    realized obs_margin: max={margins.max():+.4f}  min={margins.min():+.4f}  "
          f"#violations(>0)={n_viol}  #grazing(>-0.1)={n_graze}")
    print(f"    realized margin trace: " + " ".join(f"{m:+.2f}" for m in margins))
    print(f"    goal_dist: start={dists[0]:.3f} -> end={dists[-1]:.3f}  "
          f"(min={min(dists):.3f})")
    print(f"    quat-norm drift: max|‖q‖-1|={max(abs(q-1) for q in qnorms):.2e}  "
          f"(RK4; loss is norm-robust)")
    print(f"    SQP iters/step: mean={np.mean(iters[1:]):.2f} (steps 1+, warm-started)  "
          f"first={iters[0]}")
    traj = np.array(traj)
    # quick verdict on grazing
    grazed = (margins.max() > -0.1)
    print(f"    GRAZING VERDICT: {'YES — realized path skirts the boundary' if grazed else 'NO — realized path clears the obstacle wide (tune geometry)'}")
    return margins, traj


def stage3_fd_gate(dyn, pp, solver, t2w, h=3):
    print(f"\n[3] V3 BPTT directional-FD gate (h={h}) ...")
    x0 = jax.lax.stop_gradient(jnp.asarray(env.START))
    # non-zero policy so the gradient is non-trivial
    policy = init_policy(jax.random.PRNGKey(0), obs_dim=env.NX, out_dim=env.NX + env.NU)
    policy = {**policy, "W2": jax.random.normal(jax.random.PRNGKey(11), policy["W2"].shape) * 0.03}

    guess0 = make_initial_guess(solver, pp, x0)
    guess0 = prime_guess(solver, t2w, policy, pp, x0, guess0)

    wl = make_shac_window_loss(solver, t2w, dyn, pp, h,
                               simulate_step=env.simulate_step, task_loss=env.task_loss,
                               obs_margin=env.obs_margin)

    def scalar_loss(phi):
        return wl(phi, x0, guess0)[0]

    sl = jax.jit(scalar_loss)
    La, Lb = float(sl(policy)), float(sl(policy))
    print(f"    determinism |L-L|={abs(La-Lb):.2e}  L={La:.6e}")

    g = jax.jit(jax.grad(scalar_loss))(policy)
    jax.block_until_ready(g)
    gnorm = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g))))
    grad_dir = jax.tree.map(lambda v: v / (gnorm + 1e-30), g)

    def tree_dot(a, b):
        return float(sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))))

    def rand_unit(key):
        leaves, td = jax.tree.flatten(policy)
        ks = jax.random.split(key, len(leaves))
        parts = [jax.random.normal(k, l.shape) for k, l in zip(ks, leaves)]
        d = jax.tree.unflatten(td, parts)
        n = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(d)))
        return jax.tree.map(lambda p: p / n, d)

    def mix(key):
        r = rand_unit(key)
        m = jax.tree.map(lambda gd, rr: gd + 0.5 * rr, grad_dir, r)
        n = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(m)))
        return jax.tree.map(lambda p: p / n, m)

    ks = jax.random.split(jax.random.PRNGKey(123), 2)
    directions = [("grad", grad_dir), ("mix0", mix(ks[0])), ("mix1", mix(ks[1]))]
    EPS_SEQ = (2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3)
    print(f"    |g|={gnorm:.4e}  eps_seq={EPS_SEQ}")

    ad_vec, fd_vec = [], []
    for name, delta in directions:
        ad_dd = tree_dot(g, delta)
        vals = []
        for eps in EPS_SEQ:
            phi_p = jax.tree.map(lambda p, q: p + eps * q, policy, delta)
            phi_m = jax.tree.map(lambda p, q: p - eps * q, policy, delta)
            vals.append((float(sl(phi_p)) - float(sl(phi_m))) / (2 * eps))
        # plateau = pair with smallest consecutive relative change
        best_i, best_d = 0, float("inf")
        for i in range(len(vals) - 1):
            d = abs(vals[i + 1] - vals[i]) / (abs(vals[i + 1]) + 1e-12)
            if d < best_d:
                best_d, best_i = d, i
        chosen = vals[best_i + 1]
        rel = abs(chosen - ad_dd) / (abs(ad_dd) + 1e-9)
        ad_vec.append(ad_dd)
        fd_vec.append(chosen)
        print(f"    {name:5s}: AD={ad_dd:+.6e}  FD={chosen:+.6e}  rel={rel:.3e}  "
              f"plateau_d={best_d:.1e}  fd_seq={[f'{v:+.3e}' for v in vals]}")
    ad_vec, fd_vec = np.array(ad_vec), np.array(fd_vec)
    cos = float(ad_vec @ fd_vec / (np.linalg.norm(ad_vec) * np.linalg.norm(fd_vec) + 1e-30))
    print(f"    cos(AD,FD) over {len(directions)} dirs = {cos:.6f}  "
          f"{'PASS' if cos > 0.99 else 'CHECK'}")
    return cos


if __name__ == "__main__":
    stage0_api_parity()
    dyn, pp, solver, t2w, policy = _build()
    stage1_forward(dyn, pp, solver, t2w, policy)
    stage2_rollout(dyn, pp, solver, t2w, policy)
    stage3_fd_gate(dyn, pp, solver, t2w, h=3)
    print("\n[done] verification complete.")
