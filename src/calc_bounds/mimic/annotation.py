"""Physician annotation of judgement items (HEART history and ECG, Wells 'PE most likely',
DVT signs) that structured data cannot provide.

Export: a CSV template with one row per (case, item): ids only, no note text. The annotator
reads the note in their own MIMIC environment and fills `value`. The file contains MIMIC
identifiers, so it lives in the run's output directory, outside the repo.
Import: values are validated against each parameter's domain. A blank value or "unknown"
leaves the item Unknown (never Absent).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from calc_bounds.cohort.generate import stable_seed
from calc_bounds.mimic.truth import SPECS, MimicCase
from calc_bounds.types import (
    BoolDomain,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    ParamId,
    Value,
)

COLUMNS = [
    "case_id",
    "calculator",
    "subject_id",
    "hadm_id",
    "note_id",
    "param",
    "analysis",
    "allowed_values",
    "value",
    "annotator",
    "comment",
]
UNKNOWN = "unknown"
YES, NO = "yes", "no"


def allowed_values(spec: ParameterSpec) -> list[str]:
    match spec.domain:
        case BoolDomain():
            return [YES, NO, UNKNOWN]
        case OrdinalDomain(levels=levels):
            return [*levels, UNKNOWN]
        case NumericDomain(unit=unit):
            return [f"number in {unit}", UNKNOWN]
    raise TypeError(spec.domain)


def annotation_template(
    cases: list[MimicCase], n_per_calculator: int | None, seed: int
) -> pd.DataFrame:
    """Rows for every case with pending items; a seeded sample of `n_per_calculator` cases
    per calculator if given. `analysis` is "primary" (judgement items) or "secondary" (items
    the record lacks, used only in the secondary analysis)."""
    rows = []
    for calc_id in sorted({c.calculator for c in cases}):
        pending = [
            c
            for c in cases
            if c.calculator == calc_id and (c.needs_annotation or c.needs_secondary_annotation)
        ]
        pool = sorted(pending, key=lambda c: c.case_id)
        if n_per_calculator is not None:
            rng = np.random.default_rng(stable_seed(seed, "annotation", calc_id))
            pool = [pool[i] for i in sorted(rng.permutation(len(pool))[:n_per_calculator])]
        for c in pool:
            items = [(p, "primary") for p in c.needs_annotation]
            items += [(p, "secondary") for p in c.needs_secondary_annotation]
            for p, analysis in items:
                rows.append(
                    {
                        "case_id": c.case_id,
                        "calculator": c.calculator,
                        "subject_id": c.subject_id,
                        "hadm_id": c.hadm_id,
                        "note_id": c.note_id,
                        "param": p,
                        "analysis": analysis,
                        "allowed_values": " | ".join(allowed_values(SPECS[p])),
                        "value": "",
                        "annotator": "",
                        "comment": "",
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_value(spec: ParameterSpec, raw: str) -> Value | None:
    """Annotated text -> value in the parameter's domain; None = unknown. Raises ValueError."""
    text = raw.strip().lower()
    if text in ("", UNKNOWN):
        return None
    match spec.domain:
        case BoolDomain():
            if text in (YES, NO):
                return text == YES
        case OrdinalDomain(levels=levels):
            if text in levels:
                return levels.index(text)
        case NumericDomain(lo=lo, hi=hi):
            x = float(text)
            if lo <= x <= hi:
                return x
    raise ValueError(f"{spec.id}: {raw!r} is not one of {allowed_values(spec)}")


def read_annotations(path: Path, cases: list[MimicCase]) -> dict[str, dict[ParamId, Value]]:
    """Filled template -> {case_id: {param: value}}. Unknown items are omitted. Every error
    (unknown case, item not pending for that case, invalid value, duplicate) is reported."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    by_id = {c.case_id: c for c in cases}
    out: dict[str, dict[ParamId, Value]] = {}
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i, row in enumerate(df.itertuples(index=False), start=2):  # CSV line numbers
        case = by_id.get(row.case_id)
        if case is None:
            errors.append(f"line {i}: unknown case {row.case_id!r}")
            continue
        if row.param not in (*case.needs_annotation, *case.needs_secondary_annotation):
            errors.append(f"line {i}: {row.param!r} is not pending for {row.case_id}")
            continue
        if (row.case_id, row.param) in seen:
            errors.append(f"line {i}: duplicate {row.case_id} / {row.param}")
            continue
        seen.add((row.case_id, row.param))
        try:
            v = parse_value(SPECS[row.param], row.value)
        except ValueError as e:
            errors.append(f"line {i}: {e}")
            continue
        if v is not None:
            out.setdefault(row.case_id, {})[row.param] = v
    if errors:
        raise ValueError("invalid annotations:\n" + "\n".join(errors))
    return out
