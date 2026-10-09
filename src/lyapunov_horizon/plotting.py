"""Shared figure style: one categorical order, recessive axes, ink-coloured text."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Fixed categorical order (colour follows the model, never its rank).
COLORS = {
    "Persistence": "#2a78d6",          # blue
    "Linear (DMD)": "#eb6834",         # orange
    "Local quadratic": "#1baf7a",      # aqua
    "True solver": "#4a3aa7",          # violet
}
TRUTH = INK

# Diverging map for the field u: blue <-> neutral grey <-> red.
DIVERGING = LinearSegmentedColormap.from_list(
    "ks_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8201f"])


def color_for(name: str) -> str:
    for key, c in COLORS.items():
        if name.startswith(key):
            return c
    return INK_2


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 160,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
    })
