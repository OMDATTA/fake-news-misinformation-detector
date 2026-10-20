"""Red/black matplotlib style shared by all figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = "#e5202e"
DARK_RED = "#8c1119"
INK = "#0d0d0f"
PAPER = "#f2efea"
GREY = "#8a8580"

plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
    "axes.edgecolor": "#3a3a3d", "axes.labelcolor": PAPER, "text.color": PAPER,
    "xtick.color": GREY, "ytick.color": GREY, "grid.color": "#2a2a2d",
    "axes.grid": True, "grid.linewidth": 0.6, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})
