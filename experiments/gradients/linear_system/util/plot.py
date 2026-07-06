"""Shared plotting library for the linear_system gradient-accuracy benchmark.

ALL linear_system plotting scripts import their style constants and result
locations from this module: INK/MUTED/GRID, apply_style, save_fig,
DATA_DIR (results/data), PLOT_DIR (results/plot).
"""
from __future__ import annotations

import os

# ---- canonical result locations (this file lives in linear_system/util/) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # linear_system/
DATA_DIR = os.path.join(_PKG, "results", "data")    # CSV/JSON/NPZ metrics
PLOT_DIR = os.path.join(_PKG, "results", "plot")    # rendered figures
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

# ---------------------------------------------------------------------------
# Shared figure style — ALL linear_system plotting goes through these.
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
