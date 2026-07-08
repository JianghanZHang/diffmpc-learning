"""Training curves: hard (V1/V3) vs barrier (V2/V4) Diff-WMPC on the grazing quadrotor.

Four panels (one axis each): eval cost, train loss, eval closest constraint margin
(the elastic-sag exploit), gradient norm — all vs update. Seed 0 throughout; V3 is
the h=24 (truncation-matched) run. All style/paths come from util.plot.
Output: results/plot/quadrotor_v2_v4_training_curves.png
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
    DATA_DIR, INK, MUTED, ARM_COLORS, apply_style, save_fig, end_label,
)

# (label, csv, color, linestyle) — solid = hard / elastic-barrier, dashed = PURE barrier.
# Pure entries are skipped silently until their runs produce CSVs.
# V2/V4 = the UPDATED pure-barrier (no inequality slack) instances.
# V4's canonical *_pure_seed0.csv is being overwritten by the in-flight regsens A/B;
# _lifted.csv is the finished pure-barrier lifted run (eval 46.76->32.91).
ARMS = [
    ("V1 hard-plan",        "train_quadrotor_plan_hard_seed0.csv",          ARM_COLORS["plan_hard"], "-"),
    ("V2 barrier-plan",     "train_quadrotor_plan_barrier_pure_seed0.csv",  ARM_COLORS["plan_barrier"], "-"),
    ("V3 hard-BPTT h24",    "train_quadrotor_bptt_hard_h24_seed0.csv",      ARM_COLORS["bptt_hard"], "-"),
    ("V4 barrier-BPTT h24", "train_quadrotor_bptt_barrier_pure_seed0_lifted.csv", ARM_COLORS["bptt_barrier"], "-"),
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
    have = [(n, f, c, ls) for n, f, c, ls in ARMS
            if os.path.exists(os.path.join(DATA_DIR, f))]
    data = {name: load(f) for name, f, _, _ in have}
    colors = {name: c for name, _, c, _ in have}
    styles = {name: ls for name, _, _, ls in have}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        apply_style(ax)

    ax = axes[0, 0]
    for name, d in data.items():
        ax.plot(d["ev_u"], d["ev_c"], color=colors[name], lw=1.8, marker="o",
                ms=3.5, zorder=3, ls=styles[name])
        end_label(ax, d["ev_u"], d["ev_c"], name.split()[0])
    ax.set_title("Closed-loop eval cost (35 steps from START)", fontsize=10,
                 color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[0, 1]
    for name, d in data.items():
        ax.plot(d["upd"], np.maximum(d["loss"], 1e-3), color=colors[name], lw=1.2,
                alpha=0.85, zorder=3, ls=styles[name])
    ax.set_yscale("log")
    ax.set_title("Train loss per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 0]
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.annotate("constraint boundary (>0 = violation)", (0.02, 0.0),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=MUTED)
    for name, d in data.items():
        ax.plot(d["ev_u"], d["ev_m"], color=colors[name], lw=1.8, marker="o",
                ms=3.5, zorder=3, ls=styles[name])
        end_label(ax, d["ev_u"], d["ev_m"], name.split()[0])
    ax.set_title("Eval closest approach to obstacle (margin)", fontsize=10,
                 color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 1]
    for name, d in data.items():
        ax.plot(d["upd"], np.maximum(d["gnorm"], 1e-4), color=colors[name], lw=1.2,
                alpha=0.85, zorder=3, ls=styles[name])
    ax.set_yscale("log")
    ax.set_title("Gradient norm per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    handles = [plt.Line2D([], [], color=colors[n], lw=2.2, ls=styles[n], label=n)
               for n in data]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Diff-WMPC on the grazing quadrotor — hard vs pure-barrier arms (seed 0)",
                 fontsize=11.5, color=INK, y=1.045, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "quadrotor_v2_v4_training_curves.png")


if __name__ == "__main__":
    main()
