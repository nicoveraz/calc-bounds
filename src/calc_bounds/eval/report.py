"""Cross-source report: summaries, paired stats, attribution, S4 threshold sweep, missingness
sensitivity with natural-share reweighting, and figures. Code-only (no LLM calls)."""

from pathlib import Path

import numpy as np
import pandas as pd

from calc_bounds.bounds import from_extractions, is_determined
from calc_bounds.calculators import Calculator
from calc_bounds.cohort.generate import _priors, assign_documented, sample_truth, stable_seed
from calc_bounds.config import CohortConfig
from calc_bounds.extraction.oracle import oracle_extractions


def natural_undetermined_share(
    calc: Calculator, cohort: CohortConfig, seed: int, n: int = 2000
) -> float:
    """Share of cases undetermined from the note under the priors and missingness, without the
    coverage quota (the quota's rejection sampling keeps each stratum's conditional
    distribution, so stratum metrics can be reweighted with this share)."""
    rng = np.random.default_rng(stable_seed(seed, "natural-share", calc.id))
    priors = _priors(calc, cohort)
    und = 0
    for _ in range(n):
        truth = sample_truth(calc, priors, rng)
        doc = assign_documented(calc, truth, cohort, rng)
        und += not is_determined(
            calc, from_extractions(calc, oracle_extractions(calc.parameters, truth, doc))
        )
    return und / n


def reweight(by_cov: pd.DataFrame, shares: dict[str, float], metric: str) -> pd.DataFrame:
    """Natural-share estimate per (policy, calculator) from stratum means.
    `by_cov`: case table grouped by policy, calculator, determined_from_note."""
    rows = []
    for (policy, calc), g in by_cov.groupby(["policy", "calculator"]):
        m = g.set_index("determined_from_note")[metric]
        if True not in m.index or False not in m.index:
            continue
        p = shares[calc]
        rows.append(
            {
                "policy": policy,
                "calculator": calc,
                metric: p * m[False] + (1 - p) * m[True],
                "natural_undetermined": p,
            }
        )
    return pd.DataFrame(rows)


def reliability_figure(claims: dict[str, pd.DataFrame], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.plot([0, 1], [0, 1], color="#e4e3dc", linewidth=1)
    colors = {"haiku": "#2a78d6", "qwen_local": "#eb6834"}
    for name, df in claims.items():
        d = df[df["confidence"].notna()]
        bins = np.minimum((d["confidence"] * 10).astype(int), 9)
        g = d.groupby(bins).agg(
            conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size")
        )
        ax.plot(
            g["conf"],
            g["acc"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors.get(name, "#6b6a63"),
            label=f"{name} (raw)",
        )
    ax.set_xlabel("Confidence", color="#6b6a63")
    ax.set_ylabel("Accuracy of claim", color="#6b6a63")
    ax.set_title("Reliability of extraction confidence (raw)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
