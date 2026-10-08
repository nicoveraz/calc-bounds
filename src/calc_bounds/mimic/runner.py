"""MIMIC pipeline stages, driven by a `MimicRunConfig`.

Row-level outputs (cases, notes, extractions, traces, per-case tables) are written ONLY to
<paths.output_dir>/<run_id>/, which the config guarantees is outside the repo. `aggregate`
writes aggregate tables only (counts, rates, Wilson CIs, means), which may be committed.

  cohort      tables -> cohorts -> structured truth -> note sections -> cases.jsonl, notes.jsonl
  annotate    export the judgement-item template / import a filled one
  run         extraction (oracle or a local model) -> S1 / S3 / S4 / S3-bin, EHR as clinician
  aggregate   per-case rows -> aggregate tables
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calc_bounds.calculators import Calculator, get_calculator
from calc_bounds.cohort.generate import stable_seed
from calc_bounds.cohort.priors import DEFAULT_PRIORS
from calc_bounds.extraction import ExtractionResult
from calc_bounds.extraction.calibration import Calibrator
from calc_bounds.extraction.oracle import PrecomputedExtractor
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.mimic import cohorts
from calc_bounds.mimic import tables as T
from calc_bounds.mimic import truth as TR
from calc_bounds.mimic.annotation import annotation_template, read_annotations
from calc_bounds.mimic.clinician import EHRClinician
from calc_bounds.mimic.config import ORACLE, MimicRunConfig, assert_no_proxy_for_local
from calc_bounds.mimic.criteria import MimicCriteria, load_criteria
from calc_bounds.mimic.extract import llm_extract, local_llm, structured_oracle
from calc_bounds.mimic.notes import SectionedNote, sectioned_notes
from calc_bounds.mimic.outcomes import (
    aggregate_tables,
    case_row,
    claim_rows,
    policy_row,
    suppress_counts,
)
from calc_bounds.policies import AskAllPolicy, BoundsPolicy, Trace
from calc_bounds.policies.voi_echo import VoiEchoPolicy

FLOW_COUNTS = ["eligible", "admitted", "with_note", "selected", "pending_annotation", "fallback"]


def calculators(cfg: MimicRunConfig) -> dict[str, Calculator]:
    return {c: get_calculator(c, cfg.calculator_options.get(c)) for c in cfg.calculators}


def table_dirs(cfg: MimicRunConfig) -> T.TableDirs:
    p = cfg.paths.resolved()
    return T.TableDirs(
        hosp=p["hosp_dir"], icu=p["icu_dir"], ed=p["ed_dir"], notes_path=p["notes_path"]
    )


def _out(cfg: MimicRunConfig, *parts: str) -> Path:
    path = cfg.run_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --- Stage 1: cohort --------------------------------------------------------------------------


def _eligible(
    cfg: MimicRunConfig,
    crit: MimicCriteria,
    d: T.TableDirs,
    edstays: pd.DataFrame,
    admitted: pd.DataFrame,
    labs: pd.DataFrame,
) -> dict[str, set[int]]:
    """ED stays meeting each calculator's cohort definition (Cockcroft-Gault comes later)."""
    out: dict[str, set[int]] = {}
    calcs = set(cfg.calculators)
    hadm = set(admitted["hadm_id"])
    if "curb65" in calcs:
        out["curb65"] = cohorts.pneumonia_stays(
            edstays, T.load_diagnoses_icd(d, hadm), crit.cohorts
        )
    if "qsofa" in calcs:
        rx = T.load_prescriptions(d, hadm)
        micro = T.load_microbiologyevents(d, set(admitted["subject_id"]))
        out["qsofa"] = cohorts.suspected_infection_stays(edstays, rx, micro, crit)
    if "heart" in calcs:
        out["heart"] = cohorts.chest_pain_stays(
            edstays, T.load_triage(d), T.load_ed_diagnosis(d), crit.cohorts
        )
    for pe in {"perc", "wells_pe"} & calcs:
        out[pe] = cohorts.lab_in_window_stays(
            edstays, labs, crit.itemids.d_dimer, crit.windows.labs_hours
        )
    return out


def build_cohort(cfg: MimicRunConfig) -> Path:
    """Cases (one per ED stay x calculator) with structured truth and note sections."""
    crit = load_criteria(cfg.criteria)
    calcs = calculators(cfg)
    d = table_dirs(cfg)
    edstays = T.load_edstays(d)
    admissions = T.load_admissions(d)
    # Only admitted visits have a discharge summary (the note we read).
    admitted = edstays[edstays["hadm_id"].isin(set(admissions["hadm_id"]))]
    subjects = set(admitted["subject_id"])
    ids = crit.itemids
    labs = T.load_labevents(d, {ids.bun, ids.creatinine, ids.troponin_t, ids.d_dimer}, subjects)
    eligible = _eligible(cfg, crit, d, edstays, admitted, labs)

    pool = set().union(*eligible.values()) & set(admitted["stay_id"])
    if "cockcroft_gault" in calcs:
        pool |= set(admitted["stay_id"])  # decided once creatinine and weight are known
    stays = admitted[admitted["stay_id"].isin(pool)]
    gcs_items = {ids.gcs_eye, ids.gcs_verbal, ids.gcs_motor}
    records = TR.assemble(
        [
            TR.vitals(stays, T.load_triage(d), T.load_vitalsign(d, pool), crit),
            TR.demographics(stays, T.load_patients(d)),
            TR.labs(stays, labs, crit),
            TR.gcs(stays, T.load_chartevents(d, gcs_items, set(stays["subject_id"])), crit),
            TR.omr_values(
                stays, T.load_omr(d, {crit.omr.weight_lbs, crit.omr.bmi}, subjects), crit
            ),
            TR.comorbidities(stays, T.load_diagnoses_icd(d, set(stays["hadm_id"])), crit),
        ]
    )
    if "cockcroft_gault" in calcs:
        eligible["cockcroft_gault"] = {
            s for s, r in records.items() if {"creatinine", "weight"} <= set(r.truth)
        }
    notes = sectioned_notes(T.load_discharge_notes(d, set(stays["hadm_id"])), crit.note_sections)

    by_stay = admitted.set_index("stay_id")
    admitted_ids = set(admitted["stay_id"])
    cases: list[TR.MimicCase] = []
    flow = []
    for calc_id, calc in calcs.items():
        elig = eligible[calc_id]
        adm = sorted(elig & admitted_ids)
        with_note = [s for s in adm if int(by_stay.loc[s, "hadm_id"]) in notes]
        selected = with_note
        if cfg.max_cases_per_calculator is not None:
            rng = np.random.default_rng(stable_seed(cfg.seed, "mimic-sample", calc_id))
            keep = rng.permutation(len(with_note))[: cfg.max_cases_per_calculator]
            selected = [with_note[i] for i in sorted(keep)]
        made = []
        for s in selected:
            hadm = int(by_stay.loc[s, "hadm_id"])
            record = records.get(s, TR.StayRecord(stay_id=s, truth={}, source={}))
            made.append(
                TR.make_case(
                    calc,
                    record,
                    subject_id=int(by_stay.loc[s, "subject_id"]),
                    hadm_id=hadm,
                    note_id=notes[hadm].note_id,
                    crit=crit,
                )
            )
        cases += made
        flow.append(
            {
                "calculator": calc_id,
                "eligible": len(elig),
                "admitted": len(adm),
                "with_note": len(with_note),
                "selected": len(made),
                "pending_annotation": sum(bool(c.needs_annotation) for c in made),
                "fallback": sum(notes[c.hadm_id].fallback_full_text for c in made),
            }
        )
    used = sorted({c.hadm_id for c in cases})
    write_jsonl(_out(cfg, "cases.jsonl"), cases)
    write_jsonl(_out(cfg, "notes.jsonl"), [notes[h] for h in used])
    pd.DataFrame(flow).to_csv(_out(cfg, "cohort_flow.csv"), index=False)
    return cfg.run_dir()


SECONDARY_SUFFIX = "+secondary"


def load_cases(cfg: MimicRunConfig, secondary: bool = False) -> list[TR.MimicCase]:
    """Cases with imported physician annotations applied; secondary items only if asked."""
    cases = read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
    path = cfg.run_dir() / "annotations.json"
    if path.exists():
        ann = json.loads(path.read_text())
        cases = [
            TR.apply_annotations(c, ann.get(c.case_id, {}), secondary=secondary) for c in cases
        ]
    return cases


def rows_name(extractor: str, secondary: bool) -> str:
    return extractor + (SECONDARY_SUFFIX if secondary else "")


# --- Stage 2: annotation ----------------------------------------------------------------------


def export_annotations(cfg: MimicRunConfig, n_per_calculator: int | None) -> Path:
    cases = read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
    out = _out(cfg, "annotation_template.csv")
    annotation_template(cases, n_per_calculator, cfg.seed).to_csv(out, index=False)
    return out


def import_annotations(cfg: MimicRunConfig, filled: Path) -> Path:
    cases = read_jsonl(cfg.run_dir() / "cases.jsonl", TR.MimicCase)
    values = read_annotations(filled, cases)
    out = _out(cfg, "annotations.json")
    out.write_text(json.dumps(values, indent=1))
    return out


# --- Stage 3: run -----------------------------------------------------------------------------


def _policies(cfg: MimicRunConfig, calcs: dict[str, Calculator]) -> dict:
    """S1, S3, S3-bin and S4. S4 has no calibration on MIMIC (no labelled dev split), and its
    VOI beliefs use the synthetic cohort priors (modelling choice)."""
    made = {
        "s1_ask_all": AskAllPolicy(),
        "s3_bounds": BoundsPolicy(binary=False),
        "s3_bin": BoundsPolicy(binary=True),
        "s4_bounds_voi_echo": VoiEchoPolicy(
            {c: DEFAULT_PRIORS[c] for c in calcs}, Calibrator(method="none"), cfg.echo_threshold
        ),
    }
    return {p: made[p] for p in cfg.policies}


def extract(
    cfg: MimicRunConfig,
    extractor: str,
    cases: list[TR.MimicCase],
    notes: dict[int, SectionedNote],
) -> list[ExtractionResult]:
    calcs = calculators(cfg)
    if extractor == ORACLE:
        return [structured_oracle(calcs[c.calculator], c) for c in cases]
    x = cfg.extractors[extractor]
    llm = local_llm(cfg.providers, cfg.paths.resolved()["cache_dir"], _out(cfg, "usage.jsonl"))
    return llm_extract(
        cases,
        notes,
        calcs,
        extractor=extractor,
        provider=x.provider,
        pcfg=cfg.providers[x.provider],
        llm=llm,
        max_workers=x.max_workers,
    )


def run(cfg: MimicRunConfig, extractor: str = ORACLE, secondary: bool = False) -> Path:
    """Extraction -> policies with the EHR as the clinician -> row-level tables. With
    `secondary`, items annotated from the note (e.g. confusion) join the structured truth;
    extraction is identical (cached), only the reference and the EHR's answers change."""
    if extractor != ORACLE and extractor not in cfg.extractors:
        known = [ORACLE, *cfg.extractors]
        raise ValueError(f"unknown extractor {extractor!r}; expected one of {known}")
    assert_no_proxy_for_local()
    calcs = calculators(cfg)
    cases = load_cases(cfg, secondary)
    name = rows_name(extractor, secondary)
    notes = {n.hadm_id: n for n in read_jsonl(cfg.run_dir() / "notes.jsonl", SectionedNote)}
    results = extract(cfg, extractor, cases, notes)
    write_jsonl(_out(cfg, "extractions", f"{extractor}.jsonl"), results)
    by_case = {r.case_id: r for r in results}
    precomputed = PrecomputedExtractor(extractor, by_case)
    policies = _policies(cfg, calcs)
    traces: list[Trace] = []
    case_rows, policy_rows, claims = [], [], []
    for case in cases:
        calc, note = calcs[case.calculator], notes[case.hadm_id]
        ext = by_case[case.case_id]
        case_rows.append(case_row(case, calc, ext, note.fallback_full_text))
        claims += claim_rows(case, calc, ext)
        for policy in policies.values():
            t = policy.run(case, note.text, calc, precomputed, EHRClinician(case.truth))
            traces.append(t)
            policy_rows.append(policy_row(case, calc, t))
    write_jsonl(_out(cfg, "traces", f"{name}.jsonl"), traces)
    rows = cfg.run_dir() / "rows" / name
    rows.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_rows).to_csv(rows / "cases.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(rows / "policies.csv", index=False)
    pd.DataFrame(claims).to_csv(rows / "claims.csv", index=False)
    return rows


# --- Stage 4: aggregate -----------------------------------------------------------------------


def _read_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Booleans with missing values come back as object columns of True/False/NaN.
    for c in df.columns:
        values = set(df[c].dropna().unique())
        if values and values <= {True, False, "True", "False"}:
            df[c] = df[c].map({True: True, False: False, "True": True, "False": False})
    return df


def aggregate(cfg: MimicRunConfig, extractor: str, out: Path, secondary: bool = False) -> Path:
    """Aggregate-only tables to `out/<run_id>/<extractor>[+secondary]/` (safe to commit)."""
    name = rows_name(extractor, secondary)
    rows = cfg.run_dir() / "rows" / name
    tables = aggregate_tables(
        _read_rows(rows / "cases.csv"),
        _read_rows(rows / "policies.csv"),
        _read_rows(rows / "claims.csv"),
        cfg.min_cell_count,
    )
    flow = pd.read_csv(cfg.run_dir() / "cohort_flow.csv")
    tables["cohort_flow"] = suppress_counts(flow, FLOW_COUNTS, cfg.min_cell_count)
    dest = out / cfg.run_id / name
    dest.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(dest / f"{name}.csv", index=False)
    return dest
