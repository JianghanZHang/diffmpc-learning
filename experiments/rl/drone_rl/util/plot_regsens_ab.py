"""§4.7 regularized-sensitivity A/B on V4p pure barrier (sigma_x = 0 vs 1e-2).

Three panels: eval cost (milestones), per-update gradient norm (log; the spike
channel), and eval closest-margin — OFF vs ON. Both runs are V4p pure, lifted,
seed 0, 95 updates, identical code except the §4.7 primal Levenberg-Marquardt
term sigma_x. Output: results/plot/quadrotor_regsens_ab.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from util.plot import DATA_DIR, INK, MUTED, apply_style, save_fig, end_label  # noqa: E402

RUNS = [
    ("sigma_x = 0 (off)",   "train_quadrotor_bptt_barrier_pure_seed0_regOFF.csv",      "#c04a4a", "-"),
    ("sigma_x = 1e-2 (on)", "train_quadrotor_bptt_barrier_pure_regsx0.01sf0_seed0.csv", "#1baf7a", "-"),
]


def load(fn):
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, fn))))
    upd = np.array([int(r["update"]) for r in rows])
    g = np.array([float(r["grad_norm"]) for r in rows])
    eu = np.array([int(r["update"]) for r in rows if r["eval_cost"]])
    ec = np.array([float(r["eval_cost"]) for r in rows if r["eval_cost"]])
    em = np.array([float(r["eval_closest_margin"]) for r in rows if r["eval_cost"]])
    return dict(upd=upd, g=g, eu=eu, ec=ec, em=em)


def main():
    data = {n: load(f) for n, f, _, _ in RUNS}
    col = {n: c for n, f, c, _ in RUNS}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    for a in ax:
        apply_style(a)

    for n, d in data.items():
        ax[0].plot(d["eu"], d["ec"], color=col[n], lw=2.0, marker="o", ms=4, zorder=3)
        end_label(ax[0], d["eu"], d["ec"], n.split()[2])
    ax[0].set_title("Closed-loop eval cost (35 steps from START)", fontsize=10, color=INK, loc="left")
    ax[0].set_xlabel("update", fontsize=9, color=MUTED)

    for n, d in data.items():
        ax[1].semilogy(d["upd"], np.maximum(d["g"], 1e-3), color=col[n], lw=1.1, alpha=0.9, label=n)
    ax[1].axhline(50, color=MUTED, lw=1.0, ls="--")
    ax[1].annotate("grad = 50", (0.02, 50), xycoords=("axes fraction", "data"),
                   xytext=(0, 3), textcoords="offset points", fontsize=8, color=MUTED)
    ax[1].set_title("Gradient norm per update (log)", fontsize=10, color=INK, loc="left")
    ax[1].set_xlabel("update", fontsize=9, color=MUTED)
    ax[1].legend(frameon=False, fontsize=9, loc="upper right")

    ax[2].axhline(0.0, color=MUTED, lw=1.0, ls="--")
    for n, d in data.items():
        ax[2].plot(d["eu"], d["em"], color=col[n], lw=2.0, marker="o", ms=4, zorder=3)
        end_label(ax[2], d["eu"], d["em"], n.split()[2])
    ax[2].set_title("Eval closest approach to obstacle (margin)", fontsize=10, color=INK, loc="left")
    ax[2].set_xlabel("update", fontsize=9, color=MUTED)

    fig.suptitle("§4.7 regularized sensitivity on V4p pure barrier — sigma_x = 0 vs 1e-2 (seed 0)",
                 fontsize=11.5, color=INK, y=1.03, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "quadrotor_regsens_ab.png")


if __name__ == "__main__":
    main()
