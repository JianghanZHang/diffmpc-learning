"""Run the Diff-WMPC V1-vs-V3 hard-box sweep (drone or quadrotor) and aggregate a summary.

For each (variant in {plan_hard, bptt_hard}) x (seed), calls train(...) and collects the
returned summary dict; writes ``results/{env_tag}_training_summary.json`` with per-variant
mean/std of the final eval cost (the common metric) + per-run details.

Usage (cuDSS env, from repo root)::
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/turbompc \\
      /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python -u \\
      experiments/rl/drone_rl/run_experiment.py \\
      --env quadrotor --seeds 0 1 2 --n_updates 150
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "turbompc")
for _p in (_HERE, _SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import train as train_mod
import drone_env
import quadrotor_env

_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["drone", "quadrotor"], default="quadrotor")
    ap.add_argument("--variants", nargs="+", default=["plan_hard", "bptt_hard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_updates", type=int, default=150)
    ap.add_argument("--n_batch", type=int, default=10)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n_total", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=35)
    args = ap.parse_args()

    env = quadrotor_env if args.env == "quadrotor" else drone_env
    env_tag = env.__name__.replace("_env", "")

    print(f"=== run_experiment env={args.env} variants={args.variants} seeds={args.seeds} "
          f"n_updates={args.n_updates} ===")
    t0 = time.time()
    runs = []  # list of dicts
    for variant in args.variants:
        for seed in args.seeds:
            print(f"\n########## {variant} seed={seed} ##########")
            res = train_mod.train(
                variant, seed=seed, n_updates=args.n_updates, env=env,
                n_batch=args.n_batch, h=args.h, lr=args.lr,
                n_total=args.n_total, eval_every=args.eval_every, eval_steps=args.eval_steps,
            )
            res = {"variant": variant, "seed": seed, **res}
            runs.append(res)
            # Save incrementally so a crash mid-sweep doesn't lose finished runs.
            _write_summary(env_tag, args, runs, time.time() - t0)

    print(f"\n=== done in {time.time()-t0:.1f}s; summary written ===")


def _write_summary(env_tag, args, runs, elapsed):
    per_variant = {}
    for variant in sorted({r["variant"] for r in runs}):
        costs = [r["last_eval_cost"] for r in runs
                 if r["variant"] == variant and r["last_eval_cost"] is not None]
        if costs:
            per_variant[variant] = {
                "final_eval_cost_mean": float(np.mean(costs)),
                "final_eval_cost_std": float(np.std(costs)),
                "n_runs": len(costs),
            }
    summary = {
        "env": env_tag,
        "config": {k: getattr(args, k) for k in
                   ("n_updates", "n_batch", "h", "lr", "n_total", "eval_every", "eval_steps")},
        "elapsed_s": elapsed,
        "per_variant": per_variant,
        "runs": runs,
    }
    path = os.path.join(_RESULTS_DIR, f"{env_tag}_training_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
