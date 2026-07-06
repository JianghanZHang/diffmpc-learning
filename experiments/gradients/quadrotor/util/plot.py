"""Shared plotting library for the quadrotor gradient-quality package.

ALL plot scripts in this package import their style and result locations from
here: INK/MUTED/GRID, apply_style, save_fig, DATA_DIR (results/data) and
PLOT_DIR (results/plot). Modeled on experiments/rl/drone_rl/util/plot.py.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

# ---- package-relative result locations (this file lives in quadrotor/util/) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # experiments/gradients/quadrotor/

# Canonical result locations: CSV/JSON/NPZ under results/data, figures under results/plot.
DATA_DIR = os.path.join(_PKG, "results", "data")
PLOT_DIR = os.path.join(_PKG, "results", "plot")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared figure style — ALL quadrotor plotting goes through these.
# ---------------------------------------------------------------------------
INK, MUTED, GRID = "#1a1a2e", "#5a5a6e", "#e3e3ea"


def apply_style(ax):
    """House style: white ground, recessive grid, muted spines/ticks."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return ax


def save_fig(fig, name, *, dpi=150):
    """Save a figure into results/plot/ and return the path."""
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    return out
