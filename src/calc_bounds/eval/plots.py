"""Accuracy vs questions per system (static PNG for the paper/notes).

Each policy keeps a fixed color and marker (color follows the entity, never its rank);
points are direct-labelled, so identity never relies on color alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Paper figures use the look of classic base-R graphics: Helvetica, a full black box, outward
# ticks, y tick labels parallel to the axis, no grid, bold centred titles, boxed legends, open
# plotting symbols and the R 4 default palette (grey fills for grouped bars).
R_PALETTE = ["black", "#DF536B", "#61D04F", "#2297E6", "#28E2E5", "#CD0BBC", "#F5C710", "gray"]
R_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "black",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.titlelocation": "center",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "patch.edgecolor": "black",
    "pdf.fonttype": 42,
}
# R's pch symbols 1, 2, 0, 5, 6 (open circle, triangle, square, diamond, inverted triangle).
R_POLICY_STYLE: dict[str, tuple[str, str]] = {
    "s1_ask_all": ("black", "o"),
    "s2_llm_agent": ("#DF536B", "^"),
    "s3_bounds": ("#2297E6", "s"),
    "s4_bounds_voi_echo": ("#61D04F", "D"),
    "s3_bin": ("#CD0BBC", "v"),
}


def r_axes(ax: plt.Axes) -> None:
    """Base-R axis conventions: y tick labels parallel to the axis (las = 0)."""
    ax.tick_params(axis="y", labelrotation=90)
    for label in ax.get_yticklabels():
        label.set_verticalalignment("center")


def r_grays(n: int) -> list[str]:
    """R's gray.colors(n): grey levels from 0.3 to 0.9 (gamma 2.2)."""
    if n == 1:
        return ["#4D4D4D"]
    levels = [(0.3**2.2 + (0.9**2.2 - 0.3**2.2) * i / (n - 1)) ** (1 / 2.2) for i in range(n)]
    return ["#{0:02X}{0:02X}{0:02X}".format(round(255 * v)) for v in levels]


# Reference categorical palette (light mode), fixed per policy.
POLICY_STYLE: dict[str, tuple[str, str, str]] = {
    "s1_ask_all": ("#2a78d6", "o", "S1 ask-all"),
    "s3_bounds": ("#eb6834", "s", "S3 bounds"),
    "s3_bin": ("#1baf7a", "^", "S3-bin"),
    "s2_llm_agent": ("#eda100", "D", "S2 LLM agent"),
    "s4_bounds_voi_echo": ("#e87ba4", "v", "S4 bounds+VOI+echo"),
}
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e4e3dc"


def _style_axes(ax: plt.Axes, title: str, ymin: float = 0.0) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylim(ymin, 1.005)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


LABEL_OFFSETS = [(8, 6), (8, -12), (8, 18), (8, -24), (8, 30)]


def _plot_points(ax: plt.Axes, df: pd.DataFrame, *, labels: bool) -> None:
    order = {p: i for i, p in enumerate(df.sort_values("mean_questions")["policy"])}
    for _, row in df.iterrows():
        color, marker, name = POLICY_STYLE.get(row["policy"], (MUTED, "o", row["policy"]))
        ax.scatter(
            row["mean_questions"],
            row["accuracy"],
            s=64,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=2,
            zorder=3,
            label=name,
        )
        if labels:
            ax.annotate(
                name,
                (row["mean_questions"], row["accuracy"]),
                xytext=LABEL_OFFSETS[order[row["policy"]] % len(LABEL_OFFSETS)],
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )


def accuracy_vs_questions(overall: pd.DataFrame, per_calc: pd.DataFrame, out: Path) -> None:
    """`overall`: summary by policy; `per_calc`: summary by (policy, calculator)."""
    calcs = sorted(per_calc["calculator"].unique())
    ncols = 3
    nrows = 1 + -(-len(calcs) // ncols)
    fig = plt.figure(figsize=(11, 3.3 * nrows), facecolor="white")
    gs = fig.add_gridspec(nrows, ncols)

    # Zoom to the data (accuracy differences here are a few points), never below 0.
    ymin = max(0.0, min(overall["accuracy"].min(), per_calc["accuracy"].min()) - 0.03)
    ax = fig.add_subplot(gs[0, :])
    _plot_points(ax, overall, labels=len(overall) <= 3)  # legend carries identity beyond 3
    _style_axes(ax, "All calculators: decision-category accuracy vs mean questions per case", ymin)
    ax.set_xlabel("Mean questions per case", color=MUTED, fontsize=9)
    ax.set_ylabel("Category accuracy", color=MUTED, fontsize=9)
    ax.set_xlim(-0.2, overall["mean_questions"].max() * 1.15 + 0.3)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles, names, frameon=False, fontsize=8, loc="lower right")

    xmax = per_calc["mean_questions"].max() * 1.15 + 0.5
    for i, calc in enumerate(calcs):
        a = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _plot_points(a, per_calc[per_calc["calculator"] == calc], labels=False)
        _style_axes(a, calc, ymin)
        a.set_xlim(-0.2, xmax)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
