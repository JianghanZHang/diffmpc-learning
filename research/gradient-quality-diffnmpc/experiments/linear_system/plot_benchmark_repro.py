"""Box plots for the benchmark reproduction (benchmark_repro.npz): cos(AD, FD) vs solver tolerance, at the
VALIDATED FD eps=1e-4. Three panels tell the story:
  (1) batch-summed cos  -- the benchmark's headline quantity (sum_i grad_i), one value per seed;
  (2) per-sample cos    -- one value per problem instance (what the batch-sum hides);
  (3) batch-summed cos, per-sample outliers (cos<0.99) removed -- isolates the outlier domination.

    python research/gradient-quality-diffnmpc/experiments/linear_system/plot_benchmark_repro.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
VALID_EPS = 1e-4


def _box(ax, data, tols, title, ylabel):
    bp = ax.boxplot(data, positions=range(len(tols)), widths=0.6, showfliers=True,
                    patch_artist=True, flierprops=dict(marker="o", ms=3, mfc="0.4", mec="none", alpha=0.5))
    for b in bp["boxes"]:
        b.set(facecolor="lightblue", edgecolor="steelblue")
    for m in bp["medians"]:
        m.set(color="green", linewidth=1.6)
    ax.axhline(1.0, ls="--", color="0.5", lw=1)
    ax.set_xticks(range(len(tols)))
    ax.set_xticklabels([f"{t:.0e}" for t in tols], rotation=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("solver tolerance")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls=":", alpha=0.4)


def main():
    Z = np.load(os.path.join(_HERE, "results", "benchmark_repro.npz"))
    tols = list(Z["ad_tols"]); horizon = int(Z["horizon"]); nseeds = len(Z["seeds"])
    e = VALID_EPS
    # tightest tol on the left
    order = sorted(range(len(tols)), key=lambda i: tols[i])
    tols_o = [tols[i] for i in order]

    def col(prefix):
        return [Z[f"{prefix}|{tols[i]:.0e}|{e:.0e}"] for i in order]

    sum_data = [np.asarray(c).ravel() for c in col("sum")]              # [n_seeds] per tol
    ps_data = [np.asarray(c).ravel() for c in col("ps")]               # [n_seeds*64] per tol
    clean_data = [np.asarray(c)[~np.isnan(np.asarray(c))].ravel() for c in col("sumclean")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    _box(axes[0], sum_data, tols_o, "batch-summed cos\n(benchmark's headline metric)", "cos(AD, FD)")
    _box(axes[1], ps_data, tols_o, "per-sample cos\n(what the batch-sum hides)", "")
    _box(axes[2], clean_data, tols_o, "batch-summed cos,\nper-sample outliers removed", "")
    for ax in axes:
        ax.set_ylim(-1.05, 1.08)
    fig.suptitle(f"Benchmark reproduction (horizon {horizon}, {nseeds} seeds, validated FD eps={e:.0e}, "
                 f"fused fwd / DIRECT bwd, SQP iter=1)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(_HERE, "results", "benchmark_repro.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
