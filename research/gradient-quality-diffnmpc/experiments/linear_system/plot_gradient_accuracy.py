"""Box plots of per-sample cosine similarity (variant AD vs the common hard-box FD ground truth) across
solver tolerance, one panel per variant -- in the style of grad_box_accuracy_scp1_fdref.png.

    python research/gradient-quality-diffnmpc/experiments/linear_system/plot_gradient_accuracy.py
    python research/.../plot_gradient_accuracy.py --source benchmark   # benchmark_repro per-sample

Default (--source main): the 4-variant closed-loop experiment (gradient_accuracy.npz).
--source benchmark: the benchmark-reproduction per-sample cosine (benchmark_repro.npz). SAME plotting code.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

TOLS = ["1e-09", "1e-07", "1e-05", "1e-03", "1e-01"]         # tight -> loose (left -> right)
XLAB = [r"$10^{-9}$", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"]

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["main", "benchmark"], default="main")
ap.add_argument("--eps", type=float, default=1e-4, help="benchmark FD eps to plot")
args = ap.parse_args()

if args.source == "main":
    Z = np.load(os.path.join(_HERE, "results", "gradient_accuracy.npz"))
    keep = Z["gt_converged"].astype(bool)                   # all 64 here, but be safe
    VARIANTS = ["A noslack", "A slack", "B noslack", "B slack"]
    TITLES = {"A noslack": "A  —  hard box",
              "A slack":   r"A  —  slack (Moreau $\gamma$=1e4)",
              "B noslack": r"B  —  log-barrier ($\kappa$=1e-6)",
              "B slack":   r"B  —  barrier + slack"}
    panels = [(TITLES[v], [Z[f"{v}|{t}"][keep] for t in TOLS]) for v in VARIANTS]
    nrows, ncols, figsize = 2, 2, (11, 7)
    suptitle = ("Closed-loop gradient accuracy vs the true hard-constrained gradient "
                "(horizon 160, 50 steps, 64 samples)")
    outname = "gradient_accuracy.png"
else:
    Z = np.load(os.path.join(_HERE, "results", "benchmark_repro.npz"))
    ns, hz = len(Z["seeds"]), int(Z["horizon"])
    panels = [("hard box (per-sample)",
               [np.asarray(Z[f"ps|{t}|{args.eps:.0e}"]).ravel() for t in TOLS])]
    nrows, ncols, figsize = 1, 1, (7, 5)
    suptitle = (f"Benchmark reproduction — per-sample gradient accuracy "
                f"(horizon {hz}, {ns} seeds, validated FD eps={args.eps:.0e})")
    outname = "benchmark_repro_persample.png"

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True})
fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
for ax, (title, data) in zip(axes.flat, panels):
    ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=True,
               flierprops=dict(marker="o", markersize=3.2, markerfacecolor="0.35",
                               markeredgecolor="none", alpha=0.45),
               medianprops=dict(color="green", linewidth=2.0),
               boxprops=dict(facecolor="lightblue", edgecolor="black", linewidth=1.3),
               whiskerprops=dict(color="skyblue", linewidth=1.3),
               capprops=dict(color="skyblue", linewidth=1.3))
    ax.axhline(1.0, ls="--", color="0.5", lw=1.0)
    ax.set_xticks(range(1, len(TOLS) + 1)); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-1.06, 1.06)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.grid(axis="x", visible=False)
for ax in axes[:, 0]:
    ax.set_ylabel("Cosine similarity (AD vs FD)")
for ax in axes[-1, :]:
    ax.set_xlabel("Solver tolerance")
axes[-1, 0].legend(handles=[Patch(facecolor="lightblue", edgecolor="black", label="SQP iter = 1")],
                   loc="lower left", fontsize=10)
fig.suptitle(suptitle, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(_HERE, "results", outname)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
