"""Accuracy vs questions per system (static PNG for the paper/notes).

Each policy keeps a fixed color and marker (color follows the entity, never its rank);
points are direct-labelled, so identity never relies on color alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _plot_points(ax: plt.Axes, df: pd.DataFrame, *, labels: bool) -> None:
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
                xytext=(8, -3),
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

    ax = fig.add_subplot(gs[0, :])
    _plot_points(ax, overall, labels=True)
    _style_axes(ax, "All calculators: decision-category accuracy vs mean questions per case")
    ax.set_xlabel("Mean questions per case", color=MUTED, fontsize=9)
    ax.set_ylabel("Category accuracy", color=MUTED, fontsize=9)
    ax.set_xlim(-0.2, overall["mean_questions"].max() * 1.15 + 0.3)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles, names, frameon=False, fontsize=8, loc="lower right")

    xmax = per_calc["mean_questions"].max() * 1.15 + 0.5
    for i, calc in enumerate(calcs):
        a = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _plot_points(a, per_calc[per_calc["calculator"] == calc], labels=False)
        _style_axes(a, calc)
        a.set_xlim(-0.2, xmax)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
