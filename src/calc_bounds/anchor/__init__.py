"""MedCalc-Bench anchor: our extraction + calculator code on MedCalc-Bench Verified notes.

Source: MedCalc-Bench Verified (nsk7153/MedCalc-Bench-Verified, test split, CC-BY-SA 4.0),
stored in data/raw/medcalc/ (gitignored, never committed). See docs/ANCHOR.md for provenance,
the synthetic-data exception, and caveats.

Only calculators overlapping ours are used (HEART, CURB-65, PERC, Wells PE, Cockcroft-Gault;
qSOFA is not in MedCalc-Bench). Their "Relevant Entities" are mapped onto our parameters.
MedCalc-Bench's convention treats a missing entity as False/normal; we reproduce it only in
the `medcalc_convention` scoring, never in our tri-state pipeline.
"""

import ast
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from calc_bounds.types import ParamId, Value
from calc_bounds.units import to_canonical

CALCULATOR_NAMES: dict[str, str] = {
    "HEART Score for Major Cardiac Events": "heart",
    "CURB-65 Score for Pneumonia Severity": "curb65",
    "PERC Rule for Pulmonary Embolism": "perc",
    "Wells' Criteria for Pulmonary Embolism": "wells_pe",
    "Creatinine Clearance (Cockcroft-Gault Equation)": "cockcroft_gault",
}

_HISTORY = {"Slightly suspicious": 0, "Moderately suspicious": 1, "Highly suspicious": 2}
_ECG = {"Normal": 0, "Non-specific repolarization disturbance": 1, "Significant ST deviation": 2}
_TROPONIN = {
    "less than or equal to normal limit": 0,
    "between the normal limit or up to three times the normal limit": 1,
    "greater than three times normal limit": 2,
}
_UNIT_TOKENS = {
    "years": "years",
    "kg": "kg",
    "mg/dL": "mg/dL",
    "µmol/L": "umol/L",
    "beats per minute": "/min",
    "beats/min": "/min",
    "beats/minute": "/min",
    "bpm": "/min",
    "per min": "/min",
    "lbs": "lb",
    "breaths per minute": "/min",
    "mm hg": "mmHg",
    "%": "%",
}


class AnchorCase(BaseModel):
    row_number: int
    note_id: str
    note_type: str
    calculator: str
    note: str
    entities: dict[ParamId, Value]
    """MedCalc-Bench's annotated entities mapped to our parameters (canonical units). Params
    absent from this dict were not annotated (MedCalc treats them as False/normal)."""
    ground_truth: float
    lower: float
    upper: float


def _num(param: ParamId, v: list[Any]) -> float:
    value, unit = float(v[0]), str(v[1])
    if param == "creatinine" and unit in ("mg/L", "mmol/L"):
        # Anchor-only units (kept out of units.py so extraction prompts stay unchanged):
        # 1 mg/L = 0.1 mg/dL; 1 mmol/L = 1000 umol/L = 1000 / 88.4 mg/dL.
        return value * (0.1 if unit == "mg/L" else 1000 / 88.4)
    if param == "urea":
        # MedCalc gives BUN in mg/dL or mmol/L; BUN mmol/L equals urea mmol/L (molar).
        return to_canonical("urea", value, "bun_mg/dL" if unit == "mg/dL" else "mmol/L")
    return to_canonical(param, value, _UNIT_TOKENS[unit])


def map_entities(calc: str, raw: dict[str, Any]) -> dict[ParamId, Value]:
    e: dict[ParamId, Value] = {}

    def flag(*keys: str) -> bool | None:
        vals = [raw[k] for k in keys if k in raw]
        return any(bool(v) for v in vals) if vals else None

    if "age" in raw:
        e["age"] = _num("age", raw["age"])
    match calc:
        case "heart":
            if "Suspicion History" in raw:
                e["heart_history"] = _HISTORY[raw["Suspicion History"]]
            if "Electrocardiogram Test" in raw:
                e["heart_ecg"] = _ECG[raw["Electrocardiogram Test"]]
            if "Initial troponin" in raw:
                e["heart_troponin"] = _TROPONIN[raw["Initial troponin"]]
            for ours, theirs in [
                ("hypertension", "Hypertension history"),
                ("diabetes", "Diabetes mellitus"),
                ("hypercholesterolemia", "hypercholesterolemia"),
                ("obesity", "obesity"),
                ("smoking", "smoking"),
                (
                    "family_history_cad",
                    "parent or sibling with Cardiovascular disease before age 65",
                ),
            ]:
                if (v := flag(theirs)) is not None:
                    e[ours] = v
            if (
                v := flag("atherosclerotic disease", "Transient Ischemic Attacks History")
            ) is not None:
                e["atherosclerotic_disease"] = v
        case "curb65":
            if "Confusion" in raw:
                e["confusion"] = bool(raw["Confusion"])
            for ours, theirs in [
                ("urea", "Blood Urea Nitrogen (BUN)"),
                ("resp_rate", "respiratory rate"),
                ("sbp", "Systolic Blood Pressure"),
                ("dbp", "Diastolic Blood Pressure"),
            ]:
                if theirs in raw:
                    e[ours] = _num(ours, raw[theirs])
        case "perc" | "wells_pe":
            if "Heart Rate or Pulse" in raw:
                e["heart_rate"] = _num("heart_rate", raw["Heart Rate or Pulse"])
            if "O₂ saturation percentage" in raw:
                e["spo2"] = _num("spo2", raw["O₂ saturation percentage"])
            pairs = [
                ("hemoptysis", ("Hemoptysis",)),
                (
                    "prior_vte",
                    (
                        "Previously Documented Pulmonary Embolism",
                        "Previously documented Deep Vein Thrombosis",
                    ),
                ),
                ("unilateral_leg_swelling", ("Unilateral Leg Swelling",)),
                ("hormone_use", ("Hormone use",)),
                ("recent_surgery_trauma", ("Recent surgery or trauma",)),
                ("pe_most_likely", ("Pulmonary Embolism is #1 diagnosis OR equally likely",)),
                (
                    "immobilization_or_surgery",
                    ("Immobilization for at least 3 days", "Surgery in the previous 4 weeks"),
                ),
                ("malignancy", ("Malignancy with treatment within 6 months or palliative",)),
                ("dvt_signs", ("Clinical signs and symptoms of Deep Vein Thrombosis",)),
            ]
            for ours, theirs in pairs:
                if (v := flag(*theirs)) is not None:
                    e[ours] = v
        case "cockcroft_gault":
            if "sex" in raw:
                e["sex"] = 1 if raw["sex"] == "Female" else 0
            if "weight" in raw:
                e["weight"] = _num("weight", raw["weight"])
            if "creatinine" in raw:
                e["creatinine"] = _num("creatinine", raw["creatinine"])
    return e


def load_anchor(path: Path) -> list[AnchorCase]:
    df = pd.read_csv(path)
    df = df[df["Calculator Name"].isin(CALCULATOR_NAMES)]
    out = []
    for _, r in df.iterrows():
        calc = CALCULATOR_NAMES[r["Calculator Name"]]
        out.append(
            AnchorCase(
                row_number=int(r["Row Number"]),
                note_id=str(r["Note ID"]),
                note_type=str(r["Note Type"]),
                calculator=calc,
                note=str(r["Patient Note"]),
                entities=map_entities(calc, ast.literal_eval(r["Relevant Entities"])),
                ground_truth=float(r["Ground Truth Answer"]),
                lower=float(r["Lower Limit"]),
                upper=float(r["Upper Limit"]),
            )
        )
    return out
