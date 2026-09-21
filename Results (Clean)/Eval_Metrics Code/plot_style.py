"""Shared visual identity for every figure in Evaluation Metrics/Figures --
the same "Ocean Dusk" palette, serif/Times rcParams, and 300dpi export used
by Papers/unified_paper/figures/gen_figures.py, so a figure copied out of
this results archive into a paper looks native rather than pasted in.

Usage: `from plot_style import apply_style, C_DARK, C_TEAL, ...` at the top
of a generate_*.py script, call apply_style() once before plotting.
"""
import matplotlib.pyplot as plt

C_DARK, C_TEAL, C_GOLD, C_SAND, C_CORAL = "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"
C_GREY = "#B0BEC5"
PALETTE = [C_DARK, C_TEAL, C_GOLD, C_SAND, C_CORAL]


def apply_style():
    # Font sizes here are tuned for figures that get scaled down to roughly
    # 0.4-0.5x their source size once placed in the report (e.g. a 7in-wide
    # figure shown at 0.48\textwidth on a letter page) -- sized so labels are
    # still comfortably legible in print after that shrink, not just at
    # native resolution on screen.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 14, "legend.fontsize": 13, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
        "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    })


def save(fig, out_path):
    """Write both the given path and a .pdf sibling (for vector-quality
    inclusion in LaTeX), then close the figure."""
    import os
    fig.savefig(out_path)
    root, _ = os.path.splitext(out_path)
    fig.savefig(root + ".pdf")
