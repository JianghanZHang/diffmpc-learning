"""Shared plotting library for racing_rl (style constants, palette, helpers, paths).

ALL racing_rl plot scripts import style/paths from here (experiment-organizer layout).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_PKG, "results", "data")
PLOT_DIR = os.path.join(_PKG, "results", "plot")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

INK, MUTED, GRID = "#1a1a2e", "#5a5a6e", "#e3e3ea"
# color = estimator family; linestyle = clip (solid) vs noclip (dashed).
EST_COLORS = {"plan": "#2a78d6", "bptt": "#008300"}


def apply_style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return ax


def save_fig(fig, name, *, dpi=150):
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    return out
