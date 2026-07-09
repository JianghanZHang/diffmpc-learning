"""Racing obstacle-avoidance Diff-MPC training curves — 4 instances, mean +/- std over seeds.

Instances = {plan, bptt} estimator x {clip, noclip} gradient-clipping (all hard, no
slack). Four panels: eval cost, train loss, eval closest margin, gradient norm — vs
update. Bands are +/- std over seeds 0-2. Output: results/plot/racing_training.png
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from util.plot import DATA_DIR, INK, MUTED, EST_COLORS, apply_style, save_fig  # noqa: E402

# (label, stem (pre-_seed), color, linestyle)
INSTANCES = [
    ("plan (clip)",   "train_race_plan_hard_paper_noslack",        EST_COLORS["plan"], "-"),
    ("plan (noclip)", "train_race_plan_hard_paper_noslack_noclip", EST_COLORS["plan"], "--"),
    ("bptt (clip)",   "train_race_bptt_hard_paper_noslack",        EST_COLORS["bptt"], "-"),
    ("bptt (noclip)", "train_race_bptt_hard_paper_noslack_noclip", EST_COLORS["bptt"], "--"),
]


def _rows(path):
    return list(csv.DictReader(open(path)))


def _col(rows, key, eval_only=False):
    out = []
    for r in rows:
        v = r.get(key, "")
        if eval_only and (v == "" or v is None):
            continue
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out)


def load_instance(stem):
    """Load all seeds; return per-key (updates, mean, std) aligned on the common grid.

    'noclip' stems must not swallow the base stem's files (both share the prefix), so
    match the exact seed pattern.
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, f"{stem}_seed*.csv")))
    # exclude noclip files from the base (clip) stem
    if not stem.endswith("noclip"):
        paths = [p for p in paths if "noclip" not in os.path.basename(p)]
    seeds = [_rows(p) for p in paths]
    if not seeds:
        return None

    def full(key):
        upd = _col(seeds[0], "update")
        M = np.array([_col(s, key)[:len(upd)] for s in seeds])
        return upd, M.mean(0), M.std(0)

    def ev(key):
        eu = _col(seeds[0], "update")
        mask = [i for i, r in enumerate(seeds[0]) if r.get("eval_cost")]
        eu = eu[mask]
        vals = []
        for s in seeds:
            v = _col(s, key)
            vals.append(v[mask])
        M = np.array(vals)
        return eu, M.mean(0), M.std(0)

    return dict(n=len(seeds), loss=full("train_loss_mean"), g=full("grad_norm"),
                ec=ev("eval_cost"), em=ev("eval_closest_margin"))


def band(ax, x, m, s, color, ls, log=False):
    ax.plot(x, np.maximum(m, 1e-4) if log else m, color=color, lw=1.6, ls=ls, zorder=3)
    lo = np.maximum(m - s, 1e-4) if log else m - s
    ax.fill_between(x, lo, m + s, color=color, alpha=0.15, lw=0, zorder=2)


def main():
    data = [(lbl, load_instance(st), c, ls) for lbl, st, c, ls in INSTANCES]
    data = [(lbl, d, c, ls) for lbl, d, c, ls in data if d is not None]
    nseed = data[0][1]["n"] if data else 0

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        apply_style(ax)

    ax = axes[0, 0]
    for lbl, d, c, ls in data:
        band(ax, *d["ec"], c, ls)
    ax.set_title("Closed-loop eval cost", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[0, 1]
    for lbl, d, c, ls in data:
        band(ax, *d["loss"], c, ls, log=True)
    ax.set_yscale("log")
    ax.set_title("Train loss per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 0]
    ax.axhline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.annotate("constraint boundary (>0 = violation)", (0.02, 0.0),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    for lbl, d, c, ls in data:
        band(ax, *d["em"], c, ls)
    ax.set_title("Eval closest approach to obstacle (margin)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    ax = axes[1, 1]
    for lbl, d, c, ls in data:
        band(ax, *d["g"], c, ls, log=True)
    ax.set_yscale("log")
    ax.set_title("Gradient norm per update (log)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("update", fontsize=9, color=MUTED)

    handles = [plt.Line2D([], [], color=c, lw=2.2, ls=ls, label=lbl) for lbl, _d, c, ls in data]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Racing obstacle avoidance — Diff-MPC training (mean +/- std, {nseed} seeds)",
                 fontsize=12.5, color=INK, y=1.05, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "racing_training.png")


if __name__ == "__main__":
    main()
