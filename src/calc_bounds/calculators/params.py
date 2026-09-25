"""Shared parameter catalog. Calculators reference these specs; hidden truth, extraction and
questions are all per parameter, so a parameter used by several calculators is one entry.

Numeric params hold measured values in canonical units; calculators apply thresholds.
Plausible ranges and "stated normal" intervals (`absent_means`) are modelling choices, not
clinical criteria, and are listed for review in docs/CALCULATOR_NOTES.md.
"""

from calc_bounds.types import BoolDomain, NumericDomain, OrdinalDomain, ParameterSpec


def _bool(id: str, label: str, *, negatable: bool = True) -> ParameterSpec:
    return ParameterSpec(id=id, label=label, domain=BoolDomain(), negatable=negatable)


# --- Demographics and anthropometrics (never "absent") ------------------------------------------

AGE = ParameterSpec(
    id="age",
    label="Age (years)",
    domain=NumericDomain(unit="years", lo=18, hi=110),
    negatable=False,
)
SEX = ParameterSpec(
    id="sex", label="Sex", domain=OrdinalDomain(levels=("male", "female")), negatable=False
)
WEIGHT = ParameterSpec(
    id="weight",
    label="Body weight (kg)",
    domain=NumericDomain(unit="kg", lo=30, hi=250),
    negatable=False,
)

# --- Vital signs ------------------------------------------------------------------------------
# TODO(physician-review): "stated normal" intervals below. Each is chosen so that it does not
# straddle any calculator threshold on that vital sign (e.g. SBP normal starts above qSOFA's 100).

HEART_RATE = ParameterSpec(
    id="heart_rate",
    label="Heart rate (/min)",
    domain=NumericDomain(unit="/min", lo=20, hi=250),
    absent_means=(60, 99),
)
RESP_RATE = ParameterSpec(
    id="resp_rate",
    label="Respiratory rate (/min)",
    domain=NumericDomain(unit="/min", lo=4, hi=60),
    absent_means=(12, 20),
)
SBP = ParameterSpec(
    id="sbp",
    label="Systolic blood pressure (mmHg)",
    domain=NumericDomain(unit="mmHg", lo=50, hi=250),
    absent_means=(101, 139),
)
DBP = ParameterSpec(
    id="dbp",
    label="Diastolic blood pressure (mmHg)",
    domain=NumericDomain(unit="mmHg", lo=20, hi=150),
    absent_means=(61, 89),
)
SPO2 = ParameterSpec(
    id="spo2",
    label="Oxygen saturation on room air (%)",
    domain=NumericDomain(unit="%", lo=50, hi=100),
    absent_means=(95, 100),
)

# --- Laboratory -------------------------------------------------------------------------------

UREA = ParameterSpec(
    id="urea",
    label="Serum urea (mmol/L)",
    domain=NumericDomain(unit="mmol/L", lo=1, hi=60),
    absent_means=(2.5, 7.0),  # TODO(physician-review): "urea normal" interval
)
CREATININE = ParameterSpec(
    id="creatinine",
    label="Serum creatinine (mg/dL)",
    domain=NumericDomain(unit="mg/dL", lo=0.2, hi=15),
    # TODO(physician-review): "creatinine normal" is not precise enough for Cockcroft-Gault,
    # so creatinine can only be Present or Unknown.
    negatable=False,
)

# --- Mental status ----------------------------------------------------------------------------
# Kept as two parameters because the sources define them differently.
# TODO(physician-review): should CURB-65 confusion and qSOFA altered mentation share one param?

CONFUSION = _bool("confusion", "Confusion (AMT <= 8 or new disorientation to person/place/time)")
ALTERED_MENTATION = _bool("altered_mentation", "Altered mentation (GCS < 15)")

# --- HEART ------------------------------------------------------------------------------------

HEART_HISTORY = ParameterSpec(
    id="heart_history",
    label="HEART history (clinician judgement of anginal suspicion)",
    domain=OrdinalDomain(
        levels=("slightly_suspicious", "moderately_suspicious", "highly_suspicious")
    ),
    negatable=False,
)
HEART_ECG = ParameterSpec(
    id="heart_ecg",
    label="HEART ECG",
    domain=OrdinalDomain(
        levels=("normal", "nonspecific_repolarization", "significant_st_deviation")
    ),
)
HEART_TROPONIN = ParameterSpec(
    id="heart_troponin",
    label="Initial troponin relative to the assay's normal limit",
    domain=OrdinalDomain(levels=("le_normal", "1_to_3x_normal", "gt_3x_normal")),
)
HYPERTENSION = _bool("hypertension", "Diagnosed and/or treated hypertension")
HYPERCHOLESTEROLEMIA = _bool("hypercholesterolemia", "Diagnosed hypercholesterolaemia")
DIABETES = _bool("diabetes", "Diabetes mellitus")
OBESITY = _bool("obesity", "Obesity (BMI > 30)")
SMOKING = _bool("smoking", "Current or recent smoker")
FAMILY_HISTORY_CAD = _bool("family_history_cad", "Family history of coronary artery disease")
ATHEROSCLEROTIC_DISEASE = _bool(
    "atherosclerotic_disease",
    "Known atherosclerotic disease (prior MI, coronary revascularisation, stroke/TIA, PAD)",
)

# --- PE (PERC, Wells) -------------------------------------------------------------------------

UNILATERAL_LEG_SWELLING = _bool("unilateral_leg_swelling", "Unilateral leg swelling")
HEMOPTYSIS = _bool("hemoptysis", "Haemoptysis")
RECENT_SURGERY_TRAUMA = _bool(
    "recent_surgery_trauma", "Surgery or trauma within 4 weeks requiring hospitalisation (PERC)"
)
PRIOR_VTE = _bool("prior_vte", "Previous DVT or PE")
HORMONE_USE = _bool(
    "hormone_use", "Exogenous oestrogen use (oral contraceptives, hormone replacement)"
)
DVT_SIGNS = _bool(
    "dvt_signs", "Clinical signs of DVT (minimum: leg swelling and pain on palpation of deep veins)"
)
PE_MOST_LIKELY = _bool(
    "pe_most_likely", "Alternative diagnosis less likely than PE (clinician judgement)"
)
IMMOBILIZATION_OR_SURGERY = _bool(
    "immobilization_or_surgery", "Immobilisation > 3 days or surgery in the previous 4 weeks"
)
MALIGNANCY = _bool(
    "malignancy", "Active malignancy (on treatment, treated within 6 months, or palliative)"
)
