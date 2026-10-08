"""Structured ground truth: first ED / admission values mapped to calculator parameters.

The result is a PARTIAL assignment. A value the record does not contain is simply missing
(Unknown); in particular, the absence of an ICD code never makes a comorbidity False.

Sources per parameter (first source that yields a plausible value wins):
  vitals          triage, else the first `vitalsign` row within `windows.vitals_hours`
  age, sex        patients: age = anchor_age + year(ED arrival) - anchor_year
  urea            first BUN (mg/dL) in [arrival, arrival + labs_hours], converted by units.py
  creatinine      first creatinine (mg/dL) in the same window
  heart_troponin  first troponin T in the window, as a level relative to ref_range_upper
  confusion,      first complete GCS (ICU chartevents) in the GCS window:
  altered_mentation   total < gcs_altered_below -> True, else False
  weight          omr weight (lb) nearest to arrival within +/- weight_days, converted to kg
  obesity         omr BMI nearest to arrival (> obesity_bmi_above); else an obesity code (True)
  comorbidities   an ICD code of the admission -> True; no code -> Unknown
  judgement items (`annotation_params`): Unknown until physician annotation
Numeric values outside a parameter's plausible range are dropped and listed in `implausible`.
"""

import re

import pandas as pd
from pydantic import BaseModel

from calc_bounds.bounds import Exact, ScoreBounds, score_bounds
from calc_bounds.calculators import REGISTRY, Calculator
from calc_bounds.mimic.criteria import MimicCriteria
from calc_bounds.types import (
    BoolDomain,
    CalculatorId,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    ParamId,
    Value,
)
from calc_bounds.units import to_canonical

SPECS: dict[ParamId, ParameterSpec] = {
    p.id: p for calc in REGISTRY.values() for p in calc.parameters
}
VITALS: dict[ParamId, str] = {
    "heart_rate": "heartrate",
    "resp_rate": "resprate",
    "sbp": "sbp",
    "dbp": "dbp",
    "spo2": "o2sat",  # TODO(physician-review): triage SpO2 may be on supplemental oxygen
}
LONG = ["stay_id", "param", "value", "source"]
"""Long format used while collecting values: one row per candidate value, in priority order."""


class StayRecord(BaseModel):
    """Everything the structured record says about one ED stay (all calculators)."""

    stay_id: int
    truth: dict[ParamId, Value]
    source: dict[ParamId, str]
    implausible: list[ParamId] = []


class MimicCase(BaseModel):
    """One (ED stay, calculator) case. `truth` is partial; missing params are Unknown."""

    case_id: str
    calculator: CalculatorId
    subject_id: int
    stay_id: int
    hadm_id: int
    note_id: str
    truth: dict[ParamId, Value]
    truth_source: dict[ParamId, str]
    needs_annotation: list[ParamId]
    """Judgement items still without a value (physician annotation pending)."""
    needs_secondary_annotation: list[ParamId] = []
    """Items missing from the record, annotated from the note for the secondary analysis."""
    implausible: list[ParamId] = []


def _hours(h: float) -> pd.Timedelta:
    return pd.Timedelta(hours=h)


def _long(stay_id: pd.Series, param: ParamId, value: pd.Series, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"stay_id": stay_id.to_numpy(), "param": param, "value": value.to_numpy(), "source": source}
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=LONG)


# --- Sources ----------------------------------------------------------------------------------


def vitals(
    stays: pd.DataFrame, triage: pd.DataFrame, vitalsign: pd.DataFrame, crit: MimicCriteria
) -> pd.DataFrame:
    t = triage[triage["stay_id"].isin(stays["stay_id"])]
    v = vitalsign.merge(stays[["stay_id", "intime"]], on="stay_id")
    v = v[v["charttime"].between(v["intime"], v["intime"] + _hours(crit.windows.vitals_hours))]
    v = v.sort_values("charttime", kind="stable")
    parts = [_long(t["stay_id"], pid, t[col], "triage") for pid, col in VITALS.items()]
    parts += [_long(v["stay_id"], pid, v[col], "vitalsign") for pid, col in VITALS.items()]
    return pd.concat(parts, ignore_index=True)


def demographics(stays: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    m = stays[["stay_id", "subject_id", "intime"]].merge(patients, on="subject_id")
    age = m["anchor_age"] + m["intime"].dt.year - m["anchor_year"]
    sex = m["gender"].str.strip().str.upper().map({"M": 0, "F": 1})  # levels: male, female
    return pd.concat(
        [_long(m["stay_id"], "age", age, "patients"), _long(m["stay_id"], "sex", sex, "patients")],
        ignore_index=True,
    )


def _first_labs(stays: pd.DataFrame, labs: pd.DataFrame, itemid: int, hours: float) -> pd.DataFrame:
    m = stays[["stay_id", "subject_id", "intime"]].merge(
        labs[labs["itemid"] == itemid], on="subject_id"
    )
    m = m[m["charttime"].between(m["intime"], m["intime"] + _hours(hours))]
    return m.sort_values("charttime", kind="stable")


def _uom_is(df: pd.DataFrame, unit: str) -> pd.Series:
    return df["valueuom"].fillna("").str.strip().str.lower() == unit.lower()


_COMPARATOR = re.compile(
    r"^\s*(<|>|LESS\s+THAN|GREATER\s+THAN)\s*=?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def troponin_level(
    valuenum: float, value: str | None, upper: float, comments: str | None = None
) -> int | None:
    """Troponin level relative to the assay's upper reference limit (HEART bands as in
    Paper 1: <= ULN -> 0, 1-3x -> 1, > 3x -> 2). Censored results ("<0.01", "LESS THAN
    0.01") are used only when they settle the level; when `value` is empty, the result is
    looked for in `comments` (MIMIC stores some results there). None when undetermined.
    TODO(physician-review): troponin T assay (conventional vs hs) and ref_range_upper as the
    upper reference limit; sex-specific 99th percentiles are not modelled."""
    if pd.isna(upper) or upper <= 0:
        return None
    if not pd.isna(valuenum):
        ratio = valuenum / upper
        return 0 if ratio <= 1 else 1 if ratio <= 3 else 2
    text = value if isinstance(value, str) and value.strip() else comments
    m = _COMPARATOR.match(text if isinstance(text, str) else "")
    if m is None:
        return None
    op, x = ("<" if m.group(1)[0] in "<lL" else ">"), float(m.group(2))
    if op == "<" and x <= upper:
        return 0
    if op == ">" and x / upper >= 3:
        return 2
    return None


def labs(stays: pd.DataFrame, labevents: pd.DataFrame, crit: MimicCriteria) -> pd.DataFrame:
    h, ids = crit.windows.labs_hours, crit.itemids
    bun = _first_labs(stays, labevents, ids.bun, h)
    bun = bun[_uom_is(bun, "mg/dL")]
    urea = bun["valuenum"].map(lambda x: to_canonical("urea", x, "bun_mg/dL"))
    cr = _first_labs(stays, labevents, ids.creatinine, h)
    cr = cr[_uom_is(cr, "mg/dL")]
    trop = _first_labs(stays, labevents, ids.troponin_t, h)
    level = pd.Series(
        [
            troponin_level(n, v, u, c)
            for n, v, u, c in zip(
                trop["valuenum"],
                trop["value"],
                trop["ref_range_upper"],
                trop["comments"],
                strict=True,
            )
        ],
        dtype=object,
    )
    return pd.concat(
        [
            _long(bun["stay_id"], "urea", urea, f"labevents:{ids.bun}"),
            _long(cr["stay_id"], "creatinine", cr["valuenum"], f"labevents:{ids.creatinine}"),
            _long(trop["stay_id"], "heart_troponin", level, f"labevents:{ids.troponin_t}"),
        ],
        ignore_index=True,
    )


def gcs(stays: pd.DataFrame, chartevents: pd.DataFrame, crit: MimicCriteria) -> pd.DataFrame:
    """First charttime with all three GCS components in the window.
    TODO(physician-review): GCS < 15 as the CURB-65 confusion proxy; intubated / sedated
    patients (verbal 'No Response-ETT') are not handled specially."""
    ids = crit.itemids
    component = {ids.gcs_eye: "eye", ids.gcs_verbal: "verbal", ids.gcs_motor: "motor"}
    m = stays[["stay_id", "subject_id", "intime"]].merge(
        chartevents.drop(columns=["stay_id"], errors="ignore"), on="subject_id"
    )
    m = m[m["charttime"].between(m["intime"], m["intime"] + _hours(crit.windows.gcs_hours))]
    m = m.assign(component=m["itemid"].map(component)).dropna(subset=["component", "valuenum"])
    if m.empty:
        return _empty()
    wide = m.pivot_table(
        index=["stay_id", "charttime"], columns="component", values="valuenum", aggfunc="first"
    ).reindex(columns=["eye", "verbal", "motor"])
    wide = wide.dropna().reset_index().sort_values("charttime", kind="stable")
    first = wide.groupby("stay_id", as_index=False).first()
    altered = (first["eye"] + first["verbal"] + first["motor"]) < crit.gcs_altered_below
    return pd.concat(
        [
            _long(first["stay_id"], "altered_mentation", altered, "chartevents:gcs"),
            _long(first["stay_id"], "confusion", altered, "chartevents:gcs"),
        ],
        ignore_index=True,
    )


def omr_values(stays: pd.DataFrame, omr: pd.DataFrame, crit: MimicCriteria) -> pd.DataFrame:
    m = stays[["stay_id", "subject_id", "intime"]].merge(omr, on="subject_id")
    m = m.assign(
        num=pd.to_numeric(m["result_value"], errors="coerce"),
        dist=(m["chartdate"] - m["intime"].dt.normalize()).abs(),
    )
    m = m[m["dist"] <= pd.Timedelta(days=crit.windows.weight_days)]
    m = m.sort_values(["dist", "chartdate"], kind="stable")
    w = m[m["result_name"] == crit.omr.weight_lbs]
    b = m[m["result_name"] == crit.omr.bmi].dropna(subset=["num"])
    kg = w["num"].map(lambda x: to_canonical("weight", x, "lb"))
    return pd.concat(
        [
            _long(w["stay_id"], "weight", kg, "omr:weight"),
            _long(b["stay_id"], "obesity", b["num"] > crit.obesity_bmi_above, "omr:bmi"),
        ],
        ignore_index=True,
    )


def comorbidities(
    stays: pd.DataFrame, diagnoses_icd: pd.DataFrame, crit: MimicCriteria
) -> pd.DataFrame:
    """True where the admission carries a listed code. Never False: no code is Unknown."""
    dx = diagnoses_icd.merge(stays[["stay_id", "hadm_id"]], on="hadm_id")
    parts = []
    for pid, codes in crit.comorbidity_icd.items():
        hit = dx.loc[codes.matches(dx), "stay_id"].drop_duplicates()
        parts.append(_long(hit, pid, pd.Series(True, index=hit.index), "diagnoses_icd"))
    return pd.concat(parts, ignore_index=True) if parts else _empty()


# --- Assembly ---------------------------------------------------------------------------------


def _cast(spec: ParameterSpec, v: object) -> Value:
    match spec.domain:
        case BoolDomain():
            return bool(v)
        case OrdinalDomain():
            return int(v)  # type: ignore[call-overload]
        case NumericDomain():
            return float(v)  # type: ignore[arg-type]
    raise TypeError(spec.domain)


def _plausible(spec: ParameterSpec, v: object) -> bool:
    match spec.domain:
        case NumericDomain(lo=lo, hi=hi):
            return lo <= float(v) <= hi  # type: ignore[arg-type]
        case OrdinalDomain(levels=levels):
            return 0 <= int(v) < len(levels)  # type: ignore[call-overload]
    return True


def assemble(candidates: list[pd.DataFrame]) -> dict[int, StayRecord]:
    """Candidates in priority order -> one record per stay (first plausible value wins)."""
    nonempty = [c for c in candidates if not c.empty]
    if not nonempty:
        return {}
    long = pd.concat(nonempty, ignore_index=True)
    long = long[long["value"].notna()]
    ok = [_plausible(SPECS[p], v) for p, v in zip(long["param"], long["value"], strict=True)]
    long = long.assign(ok=ok)
    records: dict[int, StayRecord] = {}
    for stay_id, rows in long.groupby("stay_id", sort=True):
        truth: dict[ParamId, Value] = {}
        source: dict[ParamId, str] = {}
        implausible: set[ParamId] = set()
        for p, v, src, good in rows[["param", "value", "source", "ok"]].itertuples(index=False):
            if not good:
                implausible.add(p)
            elif p not in truth:
                truth[p], source[p] = _cast(SPECS[p], v), src
        records[int(stay_id)] = StayRecord(  # type: ignore[call-overload]
            stay_id=int(stay_id), truth=truth, source=source, implausible=sorted(implausible)
        )
    return records


def make_case(
    calc: Calculator,
    record: StayRecord,
    *,
    subject_id: int,
    hadm_id: int,
    note_id: str,
    crit: MimicCriteria,
) -> MimicCase:
    ids = [p.id for p in calc.parameters]
    truth = {p: record.truth[p] for p in ids if p in record.truth}
    return MimicCase(
        case_id=f"{calc.id}-{record.stay_id}",
        calculator=calc.id,
        subject_id=subject_id,
        stay_id=record.stay_id,
        hadm_id=hadm_id,
        note_id=note_id,
        truth=truth,
        truth_source={p: record.source[p] for p in truth},
        needs_annotation=[p for p in ids if p in crit.annotation_params and p not in truth],
        needs_secondary_annotation=[
            p for p in ids if p in crit.secondary_annotation_params and p not in truth
        ],
        implausible=[p for p in record.implausible if p in ids],
    )


def apply_annotations(
    case: MimicCase, values: dict[ParamId, Value], *, secondary: bool = False
) -> MimicCase:
    """Add physician-annotated judgement items to the structured truth. Secondary items
    (annotated from the note because the record lacks them) are applied only when
    `secondary` is set."""
    allowed = set(case.needs_annotation)
    if secondary:
        allowed |= set(case.needs_secondary_annotation)
    ids = {p for p in values if p in allowed or case.truth_source.get(p) == "annotation"}
    return case.model_copy(
        update={
            "truth": case.truth | {p: values[p] for p in ids},
            "truth_source": case.truth_source | dict.fromkeys(ids, "annotation"),
            "needs_annotation": [p for p in case.needs_annotation if p not in ids],
            "needs_secondary_annotation": [
                p for p in case.needs_secondary_annotation if p not in ids
            ],
        }
    )


def reference_bounds(calc: Calculator, case: MimicCase) -> ScoreBounds:
    """Bounds from the structured truth alone. Its category is the reference standard when it
    is determined; otherwise the case has no reference category."""
    return score_bounds(calc, {p: Exact(value=v) for p, v in case.truth.items()})


def reference_category(calc: Calculator, case: MimicCase) -> str | None:
    cats = reference_bounds(calc, case).categories
    return next(iter(cats)) if len(cats) == 1 else None
