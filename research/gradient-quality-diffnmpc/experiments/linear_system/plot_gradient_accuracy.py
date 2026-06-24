"""Box plots of per-sample cosine similarity (variant AD vs the common hard-box FD ground truth) across
solver tolerance, one panel per variant -- in the style of grad_box_accuracy_scp1_fdref.png.

    python research/gradient-quality-diffnmpc/experiments/linear_system/plot_gradient_accuracy.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(_HERE, "results", "gradient_accuracy.npz"))
keep = Z["gt_converged"].astype(bool)                       # all 64 here, but be safe

TOLS = ["1e-09", "1e-07", "1e-05", "1e-03", "1e-01"]         # tight -> loose (left -> right)
XLAB = [r"$10^{-9}$", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"]
VARIANTS = ["A noslack", "A slack", "B noslack", "B slack"]
TITLES = {"A noslack": "A  —  hard box",
          "A slack":   r"A  —  slack (Moreau $\gamma$=1e4)",
          "B noslack": r"B  —  log-barrier ($\kappa$=1e-6)",
          "B slack":   r"B  —  barrier + slack"}

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, v in zip(axes.flat, VARIANTS):
    data = [Z[f"{v}|{t}"][keep] for t in TOLS]
    ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=True,
               flierprops=dict(marker="o", markersize=3.2, markerfacecolor="0.35",
                               markeredgecolor="none", alpha=0.45),
               medianprops=dict(color="green", linewidth=2.0),
               boxprops=dict(facecolor="lightblue", edgecolor="black", linewidth=1.3),
               whiskerprops=dict(color="skyblue", linewidth=1.3),
               capprops=dict(color="skyblue", linewidth=1.3))
    ax.axhline(1.0, ls="--", color="0.5", lw=1.0)
    ax.set_xticks(range(1, len(TOLS) + 1)); ax.set_xticklabels(XLAB)
    ax.set_title(TITLES[v], fontsize=11)
    ax.set_ylim(-1.06, 1.06)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.grid(axis="x", visible=False)
for ax in axes[:, 0]:
    ax.set_ylabel("Cosine similarity (AD vs FD)")
for ax in axes[1, :]:
    ax.set_xlabel("Solver tolerance")
axes[1, 0].legend(handles=[Patch(facecolor="lightblue", edgecolor="black", label="SQP iter = 1")],
                  loc="lower left", fontsize=10)
fig.suptitle("Closed-loop gradient accuracy vs the true hard-constrained gradient "
             "(horizon 160, 50 steps, 64 samples)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(_HERE, "results", "gradient_accuracy.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
