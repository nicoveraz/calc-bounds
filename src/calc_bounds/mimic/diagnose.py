"""Aggregate-only diagnostics of the structured truth (no ids, no values, no text).

Why a parameter is so often Unknown in the record: e.g. troponin (which lab items exist in the
window, how often the upper reference limit is missing) and GCS (how many stays have any ICU
charting). Output is counts and shares only; counts below `min_cell_count` are suppressed.
"""

from __future__ import annotations

import re

import pandas as pd

from calc_bounds.io import read_jsonl
from calc_bounds.mimic import tables as T
from calc_bounds.mimic import truth as TR
from calc_bounds.mimic.config import MimicRunConfig
from calc_bounds.mimic.criteria import load_criteria

_COMPARATOR = re.compile(r"^\s*[<>]=?\s*[0-9]*\.?[0-9]+\s*$")


def _count(n: int, floor: int) -> str:
    return str(n) if n == 0 or n >= floor else f"<{floor}"


def troponin_report(cfg: MimicRunConfig) -> pd.DataFrame:
    """Per lab item whose label mentions troponin: stays with a result in the window, and how
    often the result has a number, a comparator ('<0.01') or an upper reference limit."""
    from calc_bounds.mimic.runner import table_dirs

    crit = load_criteria(cfg.criteria)
    d = table_dirs(cfg)
    cases = [
        c
        for c in read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
        if c.calculator == "heart"
    ]
    if not cases:
        return pd.DataFrame()
    items = T.read_table(T.table_path(d.hosp, "d_labitems"), ["itemid", "label", "fluid"])
    items = items[items["label"].str.contains("troponin", case=False, na=False)]
    stays = pd.DataFrame(
        {"stay_id": [c.stay_id for c in cases], "subject_id": [c.subject_id for c in cases]}
    ).merge(T.load_edstays(d)[["stay_id", "intime"]], on="stay_id")
    labs = T.load_labevents(d, set(items["itemid"]), set(stays["subject_id"]))
    floor, rows = cfg.min_cell_count, []
    for _, it in items.iterrows():
        m = TR._first_labs(stays, labs, int(it["itemid"]), crit.windows.labs_hours)
        first = m.drop_duplicates("stay_id")
        value = first["value"].fillna("").astype(str)
        rows.append(
            {
                "itemid": int(it["itemid"]),
                "label": it["label"],
                "heart_stays": len(stays),
                "stays_with_result": _count(len(first), floor),
                "has_valuenum": _count(int(first["valuenum"].notna().sum()), floor),
                "has_comparator": _count(int(value.str.match(_COMPARATOR).sum()), floor),
                "has_ref_range_upper": _count(int(first["ref_range_upper"].notna().sum()), floor),
            }
        )
    return pd.DataFrame(rows)


def _shape(value: str) -> str:
    """Text shape of a lab value with every digit replaced by 9 ('<0.01' -> '<9.99')."""
    return re.sub(r"[0-9]", "9", " ".join(value.split()).upper())[:40]


def troponin_value_shapes(cfg: MimicRunConfig, itemid: int) -> pd.DataFrame:
    """Shapes of the first `itemid` result's text when it has no number (counts only)."""
    from calc_bounds.mimic.runner import table_dirs

    crit = load_criteria(cfg.criteria)
    d = table_dirs(cfg)
    cases = [
        c
        for c in read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
        if c.calculator == "heart"
    ]
    stays = pd.DataFrame(
        {"stay_id": [c.stay_id for c in cases], "subject_id": [c.subject_id for c in cases]}
    ).merge(T.load_edstays(d)[["stay_id", "intime"]], on="stay_id")
    labs = T.load_labevents(d, {itemid}, set(stays["subject_id"]))
    first = TR._first_labs(stays, labs, itemid, crit.windows.labs_hours).drop_duplicates("stay_id")
    no_num = first[first["valuenum"].isna()]
    value = no_num["value"].fillna("").astype(str)
    comments = no_num["comments"].fillna("").astype(str)
    text = [
        f"value={_shape(v) or '<empty>'} | comments={_shape(c) or '<empty>'}"
        for v, c in zip(value, comments, strict=True)
    ]
    counts = pd.Series(text, dtype=str).value_counts()
    return pd.DataFrame(
        {"shape": counts.index, "stays": [_count(int(n), cfg.min_cell_count) for n in counts]}
    )


def gcs_report(cfg: MimicRunConfig) -> pd.DataFrame:
    """Per calculator: stays whose record has a GCS-based value (from ICU charting)."""
    cases = read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
    floor, rows = cfg.min_cell_count, []
    for calc in sorted({c.calculator for c in cases}):
        cs = [c for c in cases if c.calculator == calc]
        for param in ("confusion", "altered_mentation"):
            with_value = sum(param in c.truth for c in cs)
            relevant = any(param in c.truth or param in c.truth_source for c in cs)
            if calc in ("curb65", "qsofa") or relevant:
                rows.append(
                    {
                        "calculator": calc,
                        "param": param,
                        "stays": len(cs),
                        "with_value": _count(with_value, floor),
                    }
                )
    return pd.DataFrame(rows)
