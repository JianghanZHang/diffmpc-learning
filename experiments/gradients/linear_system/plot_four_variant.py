"""2x2 box plots of per-sample cosine (variant AD vs the common hard-box GT) across solver tolerance, one
panel per variant, with the BATCH-SUMMED cos median overlaid (red diamond) -- in the grad_box_accuracy
style. Reads four_variant_benchmark.npz (external/turbompc, 10 seeds).

    python experiments/linear_system/plot_four_variant.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(_HERE, "results", "four_variant_benchmark.npz"), allow_pickle=True)
ns = int(Z["nseeds"]); hz = int(Z["horizon"])

TOLS = ["1e-09", "1e-07", "1e-05", "1e-03", "1e-01"]
XLAB = [r"$10^{-9}$", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"]
VARIANTS = ["1 turbompc-hardbox", "2 turbompc-moreau", "3 barrier-noslack", "4 barrier-slack"]
TITLES = {"1 turbompc-hardbox": "turbompc — hard box (external)",
          "2 turbompc-moreau":  r"turbompc — Moreau slack ($\gamma$=1e4)",
          "3 barrier-noslack":  r"log-barrier — no slack ($\kappa$=1e-6)",
          "4 barrier-slack":    r"log-barrier — Moreau slack"}

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, v in zip(axes.flat, VARIANTS):
    ps = [np.asarray(Z[f"ps|{v}|{t}"]).ravel() for t in TOLS]          # per-sample, 640 each
    bs = [float(np.median(np.asarray(Z[f"bs|{v}|{t}"]))) for t in TOLS]  # batch-sum median per tol
    ax.boxplot(ps, patch_artist=True, widths=0.62, showfliers=True,
               flierprops=dict(marker="o", markersize=3.2, markerfacecolor="0.35",
                               markeredgecolor="none", alpha=0.4),
               medianprops=dict(color="green", linewidth=2.0),
               boxprops=dict(facecolor="lightblue", edgecolor="black", linewidth=1.3),
               whiskerprops=dict(color="skyblue", linewidth=1.3),
               capprops=dict(color="skyblue", linewidth=1.3))
    ax.plot(range(1, len(TOLS) + 1), bs, "D", color="crimson", ms=6, zorder=5)  # batch-sum median
    ax.axhline(1.0, ls="--", color="0.5", lw=1.0)
    ax.set_xticks(range(1, len(TOLS) + 1)); ax.set_xticklabels(XLAB)
    ax.set_title(TITLES[v], fontsize=11); ax.set_ylim(-1.06, 1.06)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.grid(axis="x", visible=False)
for ax in axes[:, 0]:
    ax.set_ylabel("Cosine similarity (AD vs hard-box GT)")
for ax in axes[1, :]:
    ax.set_xlabel("Solver tolerance")
axes[1, 0].legend(handles=[Patch(facecolor="lightblue", edgecolor="black", label="per-sample (640)"),
                           Line2D([], [], color="green", lw=2, label="per-sample median"),
                           Line2D([], [], marker="D", color="crimson", ls="", label="batch-sum median")],
                  loc="lower left", fontsize=9)
fig.suptitle(f"4-variant closed-loop gradient accuracy vs the true hard-constrained gradient\n"
             f"(external/turbompc, horizon {hz}, {ns} seeds, SQP iter=1)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(_HERE, "results", "four_variant_benchmark.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
