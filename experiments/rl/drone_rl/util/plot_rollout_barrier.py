"""Closed-loop xy trajectories of the trained V2/V4 (barrier) policies, with V1 (hard)
as reference. Each arm deploys ITS OWN controller (barrier arms: elastic logbarrier
layer; V1: hard layer). Output: results/plot/quadrotor_rollout_v2_v4.png

Run (cuDSS env):
  PYTHONPATH=external/diffmpc2 <venv>/python -u experiments/rl/drone_rl/util/plot_rollout_barrier.py
"""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # drone_rl/
_REPO_ROOT = os.path.normpath(os.path.join(_PKG, "../../../"))
for _p in (_PKG, os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "external", "diffmpc2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import quadrotor_env as env
from mpc_layer import make_hard_layer
from policy import make_theta_to_weights
from gradient_modes import make_initial_guess, prime_guess, shift_guess
import barrier_modes
from util.plot import (
    INK, MUTED, ARM_COLORS, apply_style, save_fig, draw_obstacle, draw_start_goal,
)

_PD = os.path.join(_PKG, "trained_policies")
N_STEPS = 50


def load_policy(name):
    d = np.load(os.path.join(_PD, name))
    return {k: jnp.asarray(d[k]) for k in d.files}


def rollout_hard(policy, t2w, dyn, pp):
    layer = make_hard_layer(dyn, pp)
    solver = layer.solver
    x = jnp.array(env.START)
    guess = prime_guess(solver, t2w, policy, pp, x,
                        make_initial_guess(solver, pp, x))
    xs = [np.asarray(x)]
    for _ in range(N_STEPS):
        w = t2w(policy, x)
        sol = solver.solve(guess, {**pp, "initial_state": x}, w)
        guess = jax.lax.stop_gradient(shift_guess(sol))
        x = jax.lax.stop_gradient(env.simulate_step(dyn, x, sol.controls[0]))
        xs.append(np.asarray(x))
    return np.stack(xs)


def rollout_barrier(policy, t2w, dyn, pp):
    solver = make_hard_layer(dyn, pp).solver
    lay = barrier_modes.make_barrier_ws_layer(solver)
    x = jnp.array(env.START)
    xs = [np.asarray(x)]
    for _ in range(N_STEPS):
        w = t2w(policy, x)
        st, co = lay.solve({**pp, "initial_state": x}, w)
        x = jax.lax.stop_gradient(env.simulate_step(dyn, x, co[0]))
        xs.append(np.asarray(x))
    return np.stack(xs)


def main():
    dyn, pp = env.build_problem_params()
    t2w = make_theta_to_weights(pp[env.QK], pp[env.RK])

    arms = []
    print("rollout V1 (hard)...")
    arms.append(("V1 hard-plan", ARM_COLORS["plan_hard"],
                 rollout_hard(load_policy("train_quadrotor_plan_hard_seed0_theta.npz"),
                              t2w, dyn, pp)))
    print("rollout V2 (barrier)...")
    arms.append(("V2 barrier-plan (final, destabilized)", ARM_COLORS["plan_barrier"],
                 rollout_barrier(load_policy("train_quadrotor_plan_barrier_seed0_theta.npz"),
                                 t2w, dyn, pp)))
    print("rollout V4 (barrier)...")
    arms.append(("V4 barrier-BPTT h24", ARM_COLORS["bptt_barrier"],
                 rollout_barrier(load_policy("train_quadrotor_bptt_barrier_seed0_theta.npz"),
                                 t2w, dyn, pp)))

    obs_c, obs_r = np.asarray(env.OBS_C), float(env.OBS_R)
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=150)
    fig.patch.set_facecolor("white")
    apply_style(ax)
    draw_obstacle(ax, obs_c, obs_r)

    for name, color, xs in arms:
        p = xs[:, :2]
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.9, zorder=3)
        ax.plot(p[:, 0], p[:, 1], "o", color=color, ms=2.6, zorder=4)
        inside = np.linalg.norm(p - obs_c, axis=1) < obs_r
        if inside.any():
            ax.plot(p[inside, 0], p[inside, 1], "o", ms=5.5, mfc="none",
                    mec="#e34948", mew=1.2, zorder=5)
        k = int(len(p) * 0.42)
        ax.annotate(name.split()[0], p[k], xytext=(6, 6),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=INK)

    draw_start_goal(ax, np.asarray(env.START)[:2], np.asarray(env.GOAL)[:2])

    handles = [plt.Line2D([], [], color=c, lw=2.2, label=n) for n, c, _ in arms]
    handles.append(plt.Line2D([], [], marker="o", mfc="none", mec="#e34948",
                              ls="none", ms=6, label="step inside obstacle"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=9, color=MUTED)
    ax.set_ylabel("y [m]", fontsize=9, color=MUTED)
    ax.set_title("Deployed closed-loop trajectories (50 steps, seed-0 policies, "
                 "each arm on its own controller)", fontsize=10.5, color=INK,
                 loc="left")
    fig.tight_layout()
    save_fig(fig, "quadrotor_rollout_v2_v4.png")


if __name__ == "__main__":
    main()
