"""Renderer-family bias: 2x2 of renderer family x extractor family on the same cases.

Outcome: claim-level extraction accuracy (`metrics.claim_correct`), paired by case and param.
Same-family effect (interaction) =
    (acc[A, notes_A] - acc[A, notes_B]) - (acc[B, notes_A] - acc[B, notes_B])
i.e. how much more each extractor gains on its own family's notes. Cases are restricted to
those whose notes passed validation in every render set. CI: case-level bootstrap.
"""

import numpy as np
import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.eval.metrics import claim_correct
from calc_bounds.extraction import ExtractionResult


def claim_frame(
    cases: dict[str, PatientCase],
    results: dict[tuple[str, str], list[ExtractionResult]],
    case_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for (extractor, render), rs in results.items():
        for r in rs:
            if r.case_id not in case_ids:
                continue
            for pid, e in r.values.items():
                rows.append(
                    {
                        "extractor": extractor,
                        "render": render,
                        "case_id": r.case_id,
                        "param": pid,
                        "correct": claim_correct(cases[r.case_id], pid, e),
                    }
                )
    return pd.DataFrame(rows)


def interaction(df: pd.DataFrame, extractors: tuple[str, str], renders: tuple[str, str]) -> float:
    acc = df.groupby(["extractor", "render"])["correct"].mean()
    (xa, xb), (ra, rb) = extractors, renders
    return float((acc[xa, ra] - acc[xa, rb]) - (acc[xb, ra] - acc[xb, rb]))


def renderer_bias(
    df: pd.DataFrame,
    extractors: tuple[str, str],
    renders: tuple[str, str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """`extractors[i]` and `renders[i]` must be the same family."""
    cell = df.groupby(["extractor", "render"])["correct"].agg(["mean", "count"]).reset_index()
    point = interaction(df, extractors, renders)
    ids = df["case_id"].unique()
    by_case = {cid: g for cid, g in df.groupby("case_id")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = pd.concat([by_case[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        boots.append(interaction(sample, extractors, renders))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_cases": len(ids),
        "cells": cell.to_dict(orient="records"),
        "same_family_interaction": point,
        "ci95": [float(lo), float(hi)],
    }
