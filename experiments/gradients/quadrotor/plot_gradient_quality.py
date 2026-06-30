"""Aggregate + plot the gradient-quality sweep produced by gradient_quality_sweep.py.

Reads the sweep CSV and renders five figures (NOTE.md E1.2/E2.1):
  1. gradient error (rel-L2 & cosine vs AD-tight) vs requested NLP tolerance, per cell.
  2. gradient error vs ADMM max_iter, per cell.
  3. gradient error vs ACHIEVED convergence_error (both sweeps pooled), per cell.
  4. gradient error vs #active constraints (the activity ladder) -- the headline RQ1 plot.
  5. cos(AD-tight, FD-tight) per cell -- does the *converged* implicit gradient match FD?

Usage:
    python examples/drone_obstacles/plot_gradient_quality.py <csv>   # or newest if omitted
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELL_ORDER = ["C1_loose_off", "C2_tight_off", "C3_loose_on", "C4_tight_on"]
CELL_COLOR = {
    "C1_loose_off": "tab:green",
    "C2_tight_off": "tab:blue",
    "C3_loose_on": "tab:orange",
    "C4_tight_on": "tab:red",
}


def _newest_csv() -> str:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    csvs = sorted(glob.glob(os.path.join(out_dir, "grad_quality_*.csv")))
    if not csvs:
        raise SystemExit(f"No CSV found in {out_dir}. Run gradient_quality_sweep.py first.")
    return csvs[-1]


def _agg(df: pd.DataFrame, by) -> pd.DataFrame:
    """Mean/std of metrics grouped by `by`."""
    g = df.groupby(by)
    out = g.agg(
        cos_AD=("cos_AD", "mean"), cos_AD_sd=("cos_AD", "std"),
        rel_l2_AD=("rel_l2_AD", "mean"), rel_l2_AD_sd=("rel_l2_AD", "std"),
        rel_l2_FD=("rel_l2_FD", "mean"),
        conv=("conv_error", "mean"), n_active=("n_active", "mean"),
        descent=("descent_AD", "mean"),
    ).reset_index()
    return out


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _newest_csv()
    df = pd.read_csv(csv_path)
    cells = [c for c in CELL_ORDER if c in df["cell"].unique()]
    print(f"Loaded {len(df)} rows from {csv_path}\ncells: {cells}\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- (1) error vs requested tolerance ---
    ax_cos, ax_rel = axes[0, 0], axes[0, 1]
    tol = df[df["sweep"] == "tol"]
    for cell in cells:
        a = _agg(tol[tol["cell"] == cell], "sweep_var").sort_values("sweep_var")
        if a.empty:
            continue
        c = CELL_COLOR[cell]
        ax_cos.plot(a["sweep_var"], a["cos_AD"], "o-", color=c, label=cell)
        ax_rel.plot(a["sweep_var"], a["rel_l2_AD"].clip(lower=1e-16), "o-", color=c, label=cell)
    for ax in (ax_cos, ax_rel):
        ax.set_xscale("log"); ax.set_xlabel("requested NLP tolerance"); ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    ax_cos.set_ylabel("cosine(g, g_AD-tight)"); ax_cos.set_title("(1) Gradient direction vs tolerance")
    ax_rel.set_yscale("log"); ax_rel.set_ylabel("rel-L2 error"); ax_rel.set_title("(1) Gradient rel-L2 vs tolerance")
    ax_cos.legend(fontsize=8)

    # --- (2) error vs ADMM max_iter ---
    ax2 = axes[0, 2]
    it = df[df["sweep"] == "admm_iter"]
    for cell in cells:
        a = _agg(it[it["cell"] == cell], "sweep_var").sort_values("sweep_var")
        if a.empty:
            continue
        ax2.plot(a["sweep_var"], a["rel_l2_AD"].clip(lower=1e-16), "o-", color=CELL_COLOR[cell], label=cell)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("ADMM max_iter"); ax2.set_ylabel("rel-L2 error")
    ax2.set_title("(2) Gradient rel-L2 vs ADMM iteration budget"); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    # --- (3) error vs ACHIEVED convergence (both sweeps pooled) ---
    ax3 = axes[1, 0]
    for cell in cells:
        d = df[df["cell"] == cell]
        ax3.scatter(d["conv_error"].clip(lower=1e-16), d["rel_l2_AD"].clip(lower=1e-16),
                    s=14, alpha=0.5, color=CELL_COLOR[cell], label=cell)
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlabel("achieved convergence_error"); ax3.set_ylabel("rel-L2 error")
    ax3.set_title("(3) Gradient error vs achieved convergence"); ax3.grid(True, alpha=0.3); ax3.legend(fontsize=8)

    # --- (4) error vs #active constraints (activity ladder) ---
    ax4 = axes[1, 1]
    for cell in cells:
        d = df[df["cell"] == cell]
        ax4.scatter(d["n_active"], d["rel_l2_AD"].clip(lower=1e-16),
                    s=14, alpha=0.5, color=CELL_COLOR[cell], label=cell)
    ax4.set_yscale("log")
    ax4.set_xlabel("# active constraints (at tight solve)"); ax4.set_ylabel("rel-L2 error")
    ax4.set_title("(4) Activity ladder: error vs #active"); ax4.grid(True, alpha=0.3); ax4.legend(fontsize=8)

    # --- (5) cos(AD-tight, FD-tight) per cell (RQ1 correctness at convergence) ---
    ax5 = axes[1, 2]
    vals, labels, colors = [], [], []
    for cell in cells:
        v = df[df["cell"] == cell]["cos_AD_FD_tight"].mean()
        vals.append(v); labels.append(cell); colors.append(CELL_COLOR[cell])
    ax5.bar(range(len(vals)), vals, color=colors)
    ax5.set_xticks(range(len(labels))); ax5.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax5.set_ylabel("cosine(g_AD-tight, g_FD-tight)")
    ax5.set_title("(5) RQ1: converged AD gradient vs finite differences")
    ax5.set_ylim(min(0.9, min(vals) - 0.01) if vals else 0.9, 1.001)
    ax5.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax5.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Gradient quality on drone obstacle avoidance — {os.path.basename(csv_path)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = csv_path.replace(".csv", ".png")
    fig.savefig(png, dpi=130)
    print(f"figure -> {png}")

    # --- text summary ---
    print("\n=== summary (mean over seeds) ===")
    summ = _agg(df, ["cell", "sweep", "sweep_var"]).sort_values(["cell", "sweep", "sweep_var"])
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summ.to_string(index=False,
              columns=["cell", "sweep", "sweep_var", "cos_AD", "rel_l2_AD", "conv", "n_active", "descent"],
              float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
