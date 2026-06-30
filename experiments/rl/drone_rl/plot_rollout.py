"""Closed-loop rollout figure for the trained V1 (plan_hard) and V3 (bptt_hard) policies on
the REAL (nonlinear) quadrotor.

V3 uses the BPTT window h=24 (the regime where truncated BPTT works — see the h-sweep:
h=8 stalls at the obstacle, h>=24 matches V1). So this figure shows that with an adequate
window BOTH estimators reach the goal around the obstacle.

Self-contained: if a policy .npz is missing it (re)trains that variant first (cuDSS), saves
it under trained_policies/, then rolls out and plots. Output: results/quadrotor_rollout_v1_v3.png.

Run (cuDSS env):
  export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  PYTHONPATH=external/turbompc /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python -u \
    experiments/rl/drone_rl/plot_rollout.py
"""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
for _p in (_HERE, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "external", "turbompc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quadrotor_env as env
import train as train_mod
from mpc_layer import make_hard_layer
from policy import make_theta_to_weights
from gradient_modes import make_initial_guess, prime_guess, shift_guess

_RD = os.path.join(_HERE, "results")
_POLICY_DIR = os.path.join(_HERE, "trained_policies")
os.makedirs(_POLICY_DIR, exist_ok=True)
SEED, N_UPDATES, N_STEPS = 0, 40, 50
V3_H = 24

# (label, variant, policy_path, train_kwargs, color). V3 trained at lr=3e-3 / h=24 (the
# h-sweep setting where truncated BPTT converges to V1); V1 reuses its existing policy.
RUNS = [
    ("V1 plan_hard (open-loop-plan)", "plan_hard",
     os.path.join(_POLICY_DIR, f"train_quadrotor_plan_hard_seed{SEED}_theta.npz"),
     {"n_batch": 10, "lr": 1e-2}, "#1f77b4"),
    (f"V3 bptt_hard (trunc. BPTT, h={V3_H})", "bptt_hard",
     os.path.join(_POLICY_DIR, f"train_quadrotor_bptt_hard_h{V3_H}_seed{SEED}_theta.npz"),
     {"h": V3_H, "lr": 3e-3}, "#ff7f0e"),
]


def ensure(variant, path, train_kw):
    if os.path.exists(path):
        print(f"[{variant}] using existing {os.path.basename(path)}")
        return
    print(f"[{variant}] policy missing -> training {N_UPDATES} updates {train_kw} (cuDSS) ...")
    train_mod.train(variant, seed=SEED, n_updates=N_UPDATES, env=env,
                    n_total=50, eval_steps=N_STEPS, eval_every=40, **train_kw)
    default = os.path.join(_POLICY_DIR, f"train_quadrotor_{variant}_seed{SEED}_theta.npz")
    if os.path.abspath(default) != os.path.abspath(path):
        os.replace(default, path)
    print(f"  saved -> {os.path.basename(path)}")


def load_policy(path):
    d = np.load(path)
    return {k: jnp.array(d[k]) for k in d.files}


def rollout(solver, t2w, dyn, pp, policy, n_steps=N_STEPS):
    x = jax.lax.stop_gradient(jnp.asarray(env.START))
    guess = prime_guess(solver, t2w, policy, pp, x, make_initial_guess(solver, pp, x))
    jsolve = jax.jit(lambda g, xx, w: solver.solve(g, {**pp, "initial_state": xx}, w))
    xs, margins = [np.asarray(x)], [float(env.obs_margin(x))]
    for _ in range(n_steps):
        w = t2w(policy, jax.lax.stop_gradient(x))
        sol = jsolve(guess, x, w); jax.block_until_ready(sol.states)
        u0 = jax.lax.stop_gradient(sol.controls[0])
        x = jax.lax.stop_gradient(env.simulate_step(dyn, x, u0))
        guess = jax.lax.stop_gradient(shift_guess(sol))
        xs.append(np.asarray(x)); margins.append(float(env.obs_margin(x)))
    return np.array(xs), np.array(margins)


def main():
    for label, variant, path, kw, _ in RUNS:
        ensure(variant, path, kw)

    dyn, pp = env.build_problem_params()
    solver = make_hard_layer(dyn, pp).solver
    t2w = make_theta_to_weights(pp[env.QK], pp[env.RK])
    rolled = []
    for label, variant, path, kw, color in RUNS:
        st, mg = rollout(solver, t2w, dyn, pp, load_policy(path))
        rolled.append((label, color, st, mg))
        print(f"[{variant}] rollout: final goal_dist={float(env.goal_dist(st[-1])):.3f}  "
              f"closest_margin={mg.max():+.3f}  n_grazing={int((mg>-0.1).sum())}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    th = np.linspace(0, 2 * np.pi, 200)
    oc, orr = np.asarray(env.OBS_C), float(env.OBS_R)
    ax[0].fill(oc[0] + orr * np.cos(th), oc[1] + orr * np.sin(th), color="gray", alpha=0.35)
    ax[0].plot(oc[0] + orr * np.cos(th), oc[1] + orr * np.sin(th), color="gray", lw=1.2, label="obstacle")
    ax[0].scatter(*np.asarray(env.START[:2]), c="green", s=90, marker="o", zorder=6, label="START")
    ax[0].scatter(*np.asarray(env.GOAL[:2]), c="red", s=120, marker="*", zorder=6, label="GOAL")
    for label, color, st, _ in rolled:
        ax[0].plot(st[:, 0], st[:, 1], "-o", ms=2.5, color=color, lw=1.8, label=label)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].set_aspect("equal")
    ax[0].set_title("Closed-loop xy trajectory (nonlinear quadrotor)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for label, color, st, _ in rolled:
        d = np.linalg.norm(st[:, :3] - np.asarray(env.GOAL[:3]), axis=1)
        ax[1].plot(d, color=color, lw=1.8, label=label)
    ax[1].axhline(0.0, ls="--", color="gray", lw=1)
    ax[1].set_xlabel("closed-loop step"); ax[1].set_ylabel("distance to goal [m]")
    ax[1].set_title(f"Goal distance vs step (V3 at h={V3_H})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    out = os.path.join(_RD, "quadrotor_rollout_v1_v3.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
