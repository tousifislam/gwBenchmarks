"""Publication-quality plot settings for Nature-style figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    "black": "#1a1a1a",
    "blue": "#0072B2",
    "red": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#7B2D8E",
    "cyan": "#56B4E9",
    "gray": "#999999",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
}

COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["pink"],
    COLORS["gray"],
]

SINGLE_COL = 3.5
DOUBLE_COL = 7.2


def apply():
    """Apply Nature-style plot settings globally."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "axes.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
        "figure.figsize": (3.5, 2.8),
    })


def figsize(cols=1, aspect=0.8):
    """Return (width, height) for a figure spanning cols Nature columns."""
    w = SINGLE_COL if cols == 1 else DOUBLE_COL
    return (w, w * aspect)
