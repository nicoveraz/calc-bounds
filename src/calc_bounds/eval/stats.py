"""Paired statistics across systems on the same cases.

Accuracy: exact McNemar test on (system A correct, system B correct) per case.
Question counts: Wilcoxon signed-rank test on per-case differences (zero differences dropped,
the scipy default "wilcox" zero method). Both are two-sided.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def paired_comparison(table: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    """`table` is metrics.case_table output (one row per policy x case)."""
    x = table[table["policy"] == a].set_index("case_id")
    y = table[table["policy"] == b].set_index("case_id")
    ids = x.index.intersection(y.index)
    x, y = x.loc[ids], y.loc[ids]
    both = int((x["correct"] & y["correct"]).sum())
    only_a = int((x["correct"] & ~y["correct"]).sum())
    only_b = int((~x["correct"] & y["correct"]).sum())
    neither = int((~x["correct"] & ~y["correct"]).sum())
    mc = mcnemar([[both, only_a], [only_b, neither]], exact=True)
    diff = (x["n_questions"] - y["n_questions"]).to_numpy()
    if np.any(diff != 0):
        w = wilcoxon(x["n_questions"], y["n_questions"])
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    else:
        w_stat, w_p = float("nan"), 1.0
    return {
        "a": a,
        "b": b,
        "n": len(ids),
        "acc_a": float(x["correct"].mean()),
        "acc_b": float(y["correct"].mean()),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "mcnemar_p": float(mc.pvalue),
        "mean_q_a": float(x["n_questions"].mean()),
        "mean_q_b": float(y["n_questions"].mean()),
        "mean_q_diff": float(diff.mean()),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


COMPARISONS: list[tuple[str, str]] = [
    ("s3_bounds", "s1_ask_all"),  # H1
    ("s3_bounds", "s2_llm_agent"),  # H2
    ("s3_bounds", "s3_bin"),  # H3
    ("s4_bounds_voi_echo", "s3_bounds"),  # H4
]


def comparisons(table: pd.DataFrame) -> pd.DataFrame:
    present = set(table["policy"])
    rows = [paired_comparison(table, a, b) for a, b in COMPARISONS if {a, b} <= present]
    return pd.DataFrame(rows)
