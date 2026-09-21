"""Renders a classic Demsar (2006) critical-difference diagram, matching the
visual style used in the second paper's Figs. 6/7 (10462_2024_10724 main
text, pages 34-35): a top CD bracket, a 1..k rank axis, methods listed in
two columns (left = better half, right = worse half) connected to their
exact rank position by a drop line, and thick horizontal bars beneath the
axis joining cliques of methods that are NOT significantly different.

Pure black-and-white, matching the reference figures exactly -- this is a
static academic-figure convention, not a themed/interactive chart, so no
palette/dark-mode handling applies here.
"""
import os

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"  # matches the reference's LaTeX/Computer Modern look

# Fonts/line weights below are tuned for print legibility once this figure is
# shrunk to roughly half its native size for the report's two-per-row
# appendix layout (~0.48\textwidth) -- not for on-screen viewing at 1:1.
TICK_FS = 15
LABEL_FS = 14
TITLE_FS = 15


def find_cliques(ranks_sorted: list, cd: float) -> list:
    """ranks_sorted: avg ranks, ascending. Returns maximal (start_idx,
    end_idx) index pairs where ranks_sorted[end] - ranks_sorted[start] <= cd
    and end > start, dropping any pair fully contained in a larger one."""
    n = len(ranks_sorted)
    candidates = []
    for i in range(n):
        j = i
        while j + 1 < n and ranks_sorted[j + 1] - ranks_sorted[i] <= cd:
            j += 1
        if j > i:
            candidates.append((i, j))

    maximal = [
        c for c in candidates
        if not any(c != other and other[0] <= c[0] and c[1] <= other[1] for other in candidates)
    ]
    return maximal


def _stack_rows(cliques: list) -> dict:
    """Greedily assigns each clique to the lowest stack row whose
    already-placed cliques don't overlap its (start,end) rank-index range."""
    rows: list[list[tuple]] = []
    assignment = {}
    for c in cliques:
        placed = False
        for r, row in enumerate(rows):
            if all(c[1] < other[0] or c[0] > other[1] for other in row):
                row.append(c)
                assignment[c] = r
                placed = True
                break
        if not placed:
            rows.append([c])
            assignment[c] = len(rows) - 1
    return assignment


def plot_cd_diagram(avg_ranks: dict, cd: float, title: str, out_path: str, k: int = None) -> None:
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in items]
    ranks = [r for _, r in items]
    n_methods = len(items)
    k = k or n_methods

    half = (n_methods + 1) // 2
    left_items = list(enumerate(items[:half]))
    right_items = list(enumerate(items[half:]))

    # Clique bars are laid out BEFORE labels so labels can start clear of
    # however many stacked bar rows there turn out to be -- in the
    # reference these bars have real vertical breathing room between them,
    # not the cramped near-touching spacing a fixed small offset gives.
    cliques = find_cliques(ranks, cd)
    stack = _stack_rows(cliques)
    clique_row_height = 0.5
    clique_top = -0.2
    n_clique_rows = (max(stack.values()) + 1) if stack else 0

    fig, ax = plt.subplots(figsize=(10.5, 1.8 + 0.55 * max(len(left_items), len(right_items))))
    label_margin = 2.6  # room for the longest label names past each axis end
    ax.set_xlim(1 - label_margin, k + label_margin)  # rank 1 on the LEFT, ascending rightward -- matches the reference
    row_height = 0.55
    n_rows = max(len(left_items), len(right_items))
    label_top = clique_top - clique_row_height * n_clique_rows - 0.25
    ax.set_ylim(label_top - row_height * (n_rows + 0.5), 1.0)
    ax.axis("off")

    # Rank axis: major integer ticks (numbered) + minor ticks at 0.2
    # intervals in between, matching the reference's finer-grained ruler.
    ax.plot([1, k], [0, 0], color="black", lw=1.4)
    minor = 0
    while minor <= k - 1:
        t = 1 + minor
        if abs(t - round(t)) < 1e-9:
            ax.plot([t, t], [0, 0.05], color="black", lw=1.4)
            ax.text(t, 0.12, str(round(t)), ha="center", va="bottom", fontsize=TICK_FS)
        else:
            ax.plot([t, t], [0, 0.025], color="black", lw=0.8)
        minor += 0.2

    # CD bracket, anchored at the best (leftmost, i.e. numerically smallest) rank
    # Anchored at the axis minimum (rank 1) -- confirmed against a zoomed
    # crop of the reference (page-34 panel a): the bracket starts exactly
    # at the "1" tick, not at the best method's rank position (those two
    # just happen to sit close together when CD is small).
    cd_start = 1
    cd_y = 0.45
    ax.plot([cd_start, cd_start + cd], [cd_y, cd_y], color="black", lw=1.8)
    ax.plot([cd_start, cd_start], [cd_y - 0.04, cd_y + 0.04], color="black", lw=1.8)
    ax.plot([cd_start + cd, cd_start + cd], [cd_y - 0.04, cd_y + 0.04], color="black", lw=1.8)
    ax.text(cd_start + cd / 2, cd_y + 0.08, "CD", ha="center", va="bottom", fontsize=TICK_FS)

    # Method labels: drop line from axis to a row, then out to the margin.
    # Left half's labels sit left of xmin, right half's labels sit right of
    # xmax -- the connector line stops short of the text (a real gap, not
    # just a small offset) so the label never overlaps/strikes through it.
    label_gap = 0.45
    for row, (name, rank) in left_items:
        y = label_top - row * row_height
        ax.plot([rank, rank], [0, y], color="black", lw=1.1)
        ax.plot([rank, 1 - 0.3], [y, y], color="black", lw=1.1)
        ax.text(1 - 0.3 - label_gap, y, name, ha="right", va="center", fontsize=LABEL_FS)
    for row, (name, rank) in right_items:
        y = label_top - row * row_height
        ax.plot([rank, rank], [0, y], color="black", lw=1.1)
        ax.plot([rank, k + 0.3], [y, y], color="black", lw=1.1)
        ax.text(k + 0.3 + label_gap, y, name, ha="left", va="center", fontsize=LABEL_FS)

    # Clique bars: thick bars joining methods not significantly different.
    # Each bar overshoots the drop lines of its two end methods by a small
    # margin rather than terminating flush on them -- the reference figures
    # do this, and it reads better: a flush end is ambiguous about whether
    # the end method is inside the group or just touching it.
    bar_overhang = 0.14
    for c in cliques:
        i, j = c
        y = clique_top - clique_row_height * stack[c]
        ax.plot([ranks[i] - bar_overhang, ranks[j] + bar_overhang], [y, y],
                color="black", lw=5.0, solid_capstyle="butt")

    ax.set_title(title, fontsize=TITLE_FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    root, _ = os.path.splitext(out_path)
    fig.savefig(root + ".pdf")
    plt.close(fig)
