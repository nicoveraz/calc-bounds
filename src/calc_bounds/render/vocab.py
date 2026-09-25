"""Rendering vocabulary: clinical settings, ordinal level descriptions, custom phrasings.

The renderer gets these as English instructions for every locale; it writes the note in the
target locale directly from this structure (never by translating another note).
"""

from calc_bounds.types import ParamId

SETTINGS: dict[str, str] = {
    "heart": "Emergency department. Adult presenting with chest pain.",
    "curb65": "Emergency department. Adult with suspected community-acquired pneumonia.",
    "qsofa": "Emergency department. Adult with suspected infection.",
    "perc": (
        "Emergency department. Adult in whom pulmonary embolism is being considered; the "
        "treating clinician's overall suspicion is low."
    ),
    "wells_pe": "Emergency department. Adult being evaluated for possible pulmonary embolism.",
    "cockcroft_gault": "Hospital ward. Adult whose medications are being reviewed for dosing.",
}

LEVELS: dict[ParamId, dict[int, str]] = {
    "heart_history": {
        0: "the clinician judges the history only slightly suspicious for acute coronary "
        "syndrome (nonspecific features); write that assessment explicitly",
        1: "the clinician judges the history moderately suspicious for acute coronary syndrome "
        "(a mix of typical and nonspecific features); write that assessment explicitly",
        2: "the clinician judges the history highly suspicious for acute coronary syndrome "
        "(predominantly typical anginal features); write that assessment explicitly",
    },
    "heart_ecg": {
        0: "ECG normal",
        1: "ECG with a nonspecific repolarization abnormality (e.g. LBBB, LVH pattern, or "
        "nonspecific ST-T changes) and no significant ST deviation",
        2: "ECG with significant ST-segment depression or elevation not explained by LBBB, "
        "LVH or digoxin",
    },
    "heart_troponin": {
        0: "initial troponin at or below the assay's upper reference limit",
        1: "initial troponin above the upper reference limit, at a stated multiple strictly "
        "between 1x and 3x the upper reference limit (e.g. '2x ULN')",
        2: "initial troponin at a stated multiple clearly more than 3x the upper reference "
        "limit (e.g. '6x ULN')",
    },
    "sex": {0: "male", 1: "female"},
}

# Bool params whose natural positive/negative phrasing is not "has X" / "does not have X".
BOOL_PHRASES: dict[ParamId, tuple[str, str]] = {
    "pe_most_likely": (
        "the clinician states that pulmonary embolism is the most likely diagnosis (an "
        "alternative diagnosis is less likely)",
        "the clinician states that an alternative diagnosis is more likely than pulmonary embolism",
    ),
    "confusion": (
        "the patient is confused (new disorientation to person, place or time, or abbreviated "
        "mental test score of 8 or less)",
        "the patient is alert and fully oriented, not confused",
    ),
    "altered_mentation": (
        "the patient has altered mentation (GCS below 15; state the GCS)",
        "the patient's mentation is normal (GCS 15)",
    ),
    "smoking": ("the patient is a current smoker", "the patient has never smoked"),
    "hormone_use": (
        "the patient takes an estrogen-containing hormone (oral contraceptive or HRT)",
        "the patient takes no hormonal therapy",
    ),
}

# How a numeric param is written; the number itself is given verbatim.
NUMERIC_PHRASES: dict[ParamId, str] = {
    "age": "age {v} years",
    "heart_rate": "heart rate {v} beats/min",
    "resp_rate": "respiratory rate {v} breaths/min",
    "sbp": "systolic blood pressure {v} mmHg (write the BP as SBP/DBP if both are stated)",
    "dbp": "diastolic blood pressure {v} mmHg (write the BP as SBP/DBP if both are stated)",
    "spo2": "oxygen saturation {v}% on room air",
    "urea": "serum urea {v} mmol/L",
    "creatinine": "serum creatinine {v} mg/dL",
    "weight": "weight {v} kg",
}

MIXED_UNIT_PHRASES: dict[str, str] = {
    "urea_mg/dL": "serum urea {v} mg/dL (reported as urea, not BUN)",
    "bun_mg/dL": "BUN (blood urea nitrogen) {v} mg/dL",
    "umol/L": "serum creatinine {v} µmol/L",
    "lb": "weight {v} lb",
}

NUMERIC_NORMAL: dict[ParamId, str] = {
    "heart_rate": "heart rate normal",
    "resp_rate": "respiratory rate normal",
    "sbp": "blood pressure normal",
    "dbp": "blood pressure normal",
    "spo2": "oxygen saturation normal on room air",
    "urea": "urea within normal limits",
}

MEDICATION_CONDITION: dict[ParamId, str] = {
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "hypercholesterolemia": "high cholesterol",
}

FORBIDDEN_TERMS = [
    "HEART score",
    "CURB-65",
    "CURB65",
    "qSOFA",
    "PERC",
    "Wells",
    "Cockcroft",
    "creatinine clearance",
    "CrCl",
]
"""The note must not name the calculator or give its result."""

# Extra guidance for DO NOT MENTION items that renderers tend to leak indirectly.
NOT_MENTIONED_HINTS: dict[ParamId, str] = {
    "confusion": "Do not describe mental status, orientation, alertness or GCS at all.",
    "altered_mentation": "Do not describe mental status, orientation, alertness or GCS at all.",
    "hormone_use": "Omit the medication list entirely (a complete medication list would imply it).",
    "prior_vte": "Omit past medical history entirely (a complete history would imply it).",
    "malignancy": "Omit past medical history entirely (a complete history would imply it).",
    "pe_most_likely": (
        "Do not give a differential diagnosis, a ranking of diagnoses, or any statement of how "
        "likely pulmonary embolism is; the plan may list investigations neutrally, without "
        "empiric treatment."
    ),
    "heart_history": "Do not characterise how typical or suspicious the chest pain is.",
    "sbp": "Do not give a blood pressure or describe it.",
    "dbp": "Do not give a blood pressure or describe it.",
}
