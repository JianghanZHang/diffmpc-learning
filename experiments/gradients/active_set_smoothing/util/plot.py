"""Shared plotting library for active_set_smoothing.

ALL figure/data output paths for this package come from here: DATA_DIR
(results/data — npz/csv metrics) and PLOT_DIR (results/plot — figures).
Drivers save figures via save_fig and write npz files into DATA_DIR.
Style: INK/MUTED/GRID + apply_style(ax) (modeled on drone_rl's util/plot.py).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # active_set_smoothing/

# Canonical result locations: npz/csv under results/data, figures under results/plot.
DATA_DIR = os.path.join(_PKG, "results", "data")
PLOT_DIR = os.path.join(_PKG, "results", "plot")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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


def save_fig(fig, name, *, dpi=130):
    """Save a figure into results/plot/ and return the path."""
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    return out
