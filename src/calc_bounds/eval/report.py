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
    """Reliability diagram of raw extraction confidence (base-R look, as the paper figures).
    Writes `out` (PNG) and the same figure as PDF next to it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from calc_bounds.eval.plots import R_RC, r_axes

    names = {"haiku": "Claude Haiku 4.5", "qwen_local": "Qwen3.5-9B"}
    styles = {"haiku": ("#2297E6", "o", "-"), "qwen_local": ("#DF536B", "^", "--")}
    with matplotlib.rc_context(R_RC):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
        for name, df in claims.items():
            d = df[df["confidence"].notna()]
            bins = np.minimum((d["confidence"] * 10).astype(int), 9)
            g = d.groupby(bins).agg(
                conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size")
            )
            color, marker, line = styles.get(name, ("black", "s", "-"))
            ax.plot(
                g["conf"],
                g["acc"],
                marker=marker,
                markerfacecolor="none",
                linestyle=line,
                linewidth=1,
                markersize=5,
                color=color,
                label=names.get(name, name),
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy of claim")
        ax.set_title("Reliability of raw extraction confidence")
        ax.legend(fontsize=8, loc="center left")
        r_axes(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
