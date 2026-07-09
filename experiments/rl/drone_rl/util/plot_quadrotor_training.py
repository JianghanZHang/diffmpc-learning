"""Quadrotor obstacle-avoidance Diff-MPC training curves.

Four panels (one axis each): eval cost, train loss, eval closest constraint margin,
gradient norm — all vs update. Seed 0. Arms: hard vs pure-barrier estimators (plan /
BPTT-h24) plus the pure-barrier BPTT arm with the §4.7 regularized sensitivity.
All style/paths come from util.plot. Output: results/plot/quadrotor_training.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # drone_rl/
from util.plot import (  # noqa: E402
    DATA_DIR, INK, MUTED, ARM_COLORS, apply_style, save_fig,
)

# (legend label, end-of-line short label, csv, color, linestyle). Barrier arms are
# the pure (no-inequality-slack) instances; "+ reg-sens" is that same pure BPTT arm
# with sigma_x=1e-2 (§4.7). Missing CSVs are skipped silently.
ARMS = [
    ("hard-plan",              "hard-plan",  "train_quadrotor_plan_hard_seed0.csv",                     ARM_COLORS["plan_hard"], "-"),
    ("barrier-plan",           "barr-plan",  "train_quadrotor_plan_barrier_pure_seed0.csv",             ARM_COLORS["plan_barrier"], "-"),
    ("hard-BPTT",              "hard-BPTT",  "train_quadrotor_bptt_hard_h24_seed0.csv",                 ARM_COLORS["bptt_hard"], "-"),
    ("barrier-BPTT",           "barr-BPTT",  "train_quadrotor_bptt_barrier_pure_seed0_regOFF.csv",      ARM_COLORS["bptt_barrier"], "-"),
    ("barrier-BPTT + reg-sens", "+reg-sens", "train_quadrotor_bptt_barrier_pure_regsx0.01sf0_seed0.csv", ARM_COLORS["bptt_barrier_regsens"], "-"),
]


def load(fname):
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, fname))))
    upd = np.array([int(r["update"]) for r in rows])
    loss = np.array([float(r["train_loss_mean"]) for r in rows])
    gnorm = np.array([float(r["grad_norm"]) for r in rows])
    ev_u = np.array([int(r["update"]) for r in rows if r["eval_cost"]])
    ev_c = np.array([float(r["eval_cost"]) for r in rows if r["eval_cost"]])
    ev_m = np.array([float(r["eval_closest_margin"]) for r in rows if r["eval_cost"]])
    return dict(upd=upd, loss=loss, gnorm=gnorm, ev_u=ev_u, ev_c=ev_c, ev_m=ev_m)


def main():
    have = [a for a in ARMS if os.path.exists(os.path.join(DATA_DIR, a[2]))]
    data = {n: load(f) for n, _s, f, _c, _ls in have}
    colors = {n: c for n, _s, _f, c, _ls in have}
    styles = {n: ls for n, _s, _f, _c, ls in have}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        apply_style(ax)

    ax = axes[0, 0]
    for n, d in data.items():
        ax.plot(d["ev_u"], d["ev_c"], color=colors[n], lw=1.8, marker="o",
                ms=3.5, zorder=3, ls=styles[n])
    ax.set_title("Closed-loop eval cost (35 steps from START)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[0, 1]
    for n, d in data.items():
        ax.plot(d["upd"], np.maximum(d["loss"], 1e-3), color=colors[n], lw=1.2,
                alpha=0.85, zorder=3, ls=styles[n])
    ax.set_yscale("log")
    ax.set_title("Train loss per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 0]
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.annotate("constraint boundary (>0 = violation)", (0.02, 0.0),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=MUTED)
    for n, d in data.items():
        ax.plot(d["ev_u"], d["ev_m"], color=colors[n], lw=1.8, marker="o",
                ms=3.5, zorder=3, ls=styles[n])
    ax.set_title("Eval closest approach to obstacle (margin)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 1]
    for n, d in data.items():
        ax.plot(d["upd"], np.maximum(d["gnorm"], 1e-4), color=colors[n], lw=1.2,
                alpha=0.85, zorder=3, ls=styles[n])
    ax.set_yscale("log")
    ax.set_title("Gradient norm per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    handles = [plt.Line2D([], [], color=colors[n], lw=2.2, ls=styles[n], label=n) for n in data]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Quadrotor obstacle avoidance — Diff-MPC training (seed 0)",
                 fontsize=12.5, color=INK, y=1.05, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "quadrotor_training.png")


if __name__ == "__main__":
    main()
