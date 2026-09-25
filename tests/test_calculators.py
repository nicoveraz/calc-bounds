"""Hand-worked calculator examples from the criteria in the primary sources.

Where the source gives no worked example, cases are hand-computed from its stated criteria and
chosen to sit on thresholds (>= vs >), since that is where implementations disagree.
"""

import pytest

from calc_bounds.calculators import REGISTRY, get_calculator
from calc_bounds.calculators.cockcroft_gault import make_cockcroft_gault
from calc_bounds.types import Value

# --- HEART (Six et al. 2008; Backus et al. 2010/2013) -------------------------------------------

HEART_ZERO: dict[str, Value] = {
    "heart_history": 0,
    "heart_ecg": 0,
    "age": 30,
    "hypertension": False,
    "hypercholesterolemia": False,
    "diabetes": False,
    "obesity": False,
    "smoking": False,
    "family_history_cad": False,
    "atherosclerotic_disease": False,
    "heart_troponin": 0,
}


def heart(**kw: Value) -> tuple[float, str]:
    return REGISTRY["heart"].evaluate({**HEART_ZERO, **kw})


def test_heart_six_2008_example_30yo_man_scores_0() -> None:
    # Six 2008: 30-year-old man, non-specific pain, normal ECG and troponin -> 0.
    assert heart() == (0, "low")


@pytest.mark.parametrize(("age", "points"), [(44, 0), (44.9, 0), (45, 1), (64, 1), (65, 2)])
def test_heart_age_bands(age: float, points: int) -> None:
    assert heart(age=age)[0] == points


def test_heart_risk_factor_counts() -> None:
    assert heart(hypertension=True)[0] == 1
    assert heart(hypertension=True, diabetes=True)[0] == 1
    assert heart(hypertension=True, diabetes=True, smoking=True)[0] == 2
    assert heart(**{k: True for k in ["hypertension", "diabetes", "smoking", "obesity"]})[0] == 2
    # Atherosclerotic disease scores 2 irrespective of the other risk factors.
    assert heart(atherosclerotic_disease=True)[0] == 2
    assert heart(atherosclerotic_disease=True, hypertension=True)[0] == 2


def test_heart_categories() -> None:
    # H2 E1 A1 (50y) R1 T0 = 5 -> moderate.
    assert heart(heart_history=2, heart_ecg=1, age=50, diabetes=True) == (5, "moderate")
    # H1 E0 A0 R1 T1 = 3 -> low (upper edge).
    assert heart(heart_history=1, hypertension=True, heart_troponin=1) == (3, "low")
    # H1 E1 A1 R0 T1 = 4 -> moderate (lower edge).
    assert heart(heart_history=1, heart_ecg=1, age=50, heart_troponin=1) == (4, "moderate")
    # H2 E2 A2 R1 T0 = 7 -> high (lower edge).
    assert heart(heart_history=2, heart_ecg=2, age=70, smoking=True) == (7, "high")
    # Maximum: 10.
    assert heart(
        heart_history=2, heart_ecg=2, age=80, atherosclerotic_disease=True, heart_troponin=2
    ) == (10, "high")


# --- CURB-65 (Lim et al. 2003) ------------------------------------------------------------------

CURB_ZERO: dict[str, Value] = {
    "confusion": False,
    "urea": 5.0,
    "resp_rate": 18,
    "sbp": 120,
    "dbp": 80,
    "age": 40,
}


def curb(**kw: Value) -> tuple[float, str]:
    return REGISTRY["curb65"].evaluate({**CURB_ZERO, **kw})


def test_curb65_thresholds() -> None:
    assert curb() == (0, "low")
    assert curb(urea=7.0)[0] == 0  # > 7, not >=
    assert curb(urea=7.01)[0] == 1
    assert curb(resp_rate=29)[0] == 0
    assert curb(resp_rate=30)[0] == 1  # >= 30
    assert curb(sbp=90)[0] == 0  # < 90
    assert curb(sbp=89)[0] == 1
    assert curb(dbp=61)[0] == 0
    assert curb(dbp=60)[0] == 1  # <= 60
    assert curb(sbp=85, dbp=55)[0] == 1  # BP criterion counts once
    assert curb(age=64)[0] == 0
    assert curb(age=65)[0] == 1


def test_curb65_groups() -> None:
    assert curb(confusion=True) == (1, "low")
    assert curb(confusion=True, age=80) == (2, "moderate")
    assert curb(confusion=True, age=80, resp_rate=32) == (3, "high")
    assert curb(confusion=True, age=80, resp_rate=32, urea=12, sbp=80) == (5, "high")


# --- qSOFA (Seymour et al. 2016; Singer et al. 2016) ---

QSOFA_ZERO: dict[str, Value] = {"resp_rate": 16, "altered_mentation": False, "sbp": 120}


def qsofa(**kw: Value) -> tuple[float, str]:
    return REGISTRY["qsofa"].evaluate({**QSOFA_ZERO, **kw})


def test_qsofa() -> None:
    assert qsofa() == (0, "negative")
    assert qsofa(resp_rate=21)[0] == 0
    assert qsofa(resp_rate=22) == (1, "negative")  # >= 22
    assert qsofa(sbp=101)[0] == 0
    assert qsofa(sbp=100)[0] == 1  # <= 100
    assert qsofa(resp_rate=22, sbp=100) == (2, "positive")
    assert qsofa(resp_rate=30, sbp=90, altered_mentation=True) == (3, "positive")


# --- PERC (Kline et al. 2004, 2008) ---

PERC_ZERO: dict[str, Value] = {
    "age": 30,
    "heart_rate": 80,
    "spo2": 98,
    "unilateral_leg_swelling": False,
    "hemoptysis": False,
    "recent_surgery_trauma": False,
    "prior_vte": False,
    "hormone_use": False,
}


def perc(**kw: Value) -> tuple[float, str]:
    return REGISTRY["perc"].evaluate({**PERC_ZERO, **kw})


def test_perc() -> None:
    assert perc() == (0, "negative")
    assert perc(age=49)[1] == "negative"
    assert perc(age=50) == (1, "positive")  # rule requires age < 50
    assert perc(heart_rate=99)[1] == "negative"
    assert perc(heart_rate=100)[1] == "positive"  # requires HR < 100
    assert perc(spo2=95)[1] == "negative"  # 2008: SaO2 >= 95%
    assert perc(spo2=94.9)[1] == "positive"
    assert perc(hormone_use=True, hemoptysis=True) == (2, "positive")


# --- Wells PE, two-tier (Wells et al. 2000; van Belle et al. 2006) ---

WELLS_ZERO: dict[str, Value] = {
    "dvt_signs": False,
    "pe_most_likely": False,
    "heart_rate": 80,
    "immobilization_or_surgery": False,
    "prior_vte": False,
    "hemoptysis": False,
    "malignancy": False,
}


def wells(**kw: Value) -> tuple[float, str]:
    return REGISTRY["wells_pe"].evaluate({**WELLS_ZERO, **kw})


def test_wells_points() -> None:
    assert wells() == (0, "pe_unlikely")
    assert wells(heart_rate=100)[0] == 0  # > 100
    assert wells(heart_rate=101)[0] == 1.5
    assert wells(dvt_signs=True, pe_most_likely=True)[0] == 6
    assert wells(hemoptysis=True, malignancy=True)[0] == 2
    assert wells(immobilization_or_surgery=True, prior_vte=True)[0] == 3


def test_wells_two_tier_cutoff() -> None:
    # 4.0 -> unlikely (<= 4); 4.5 -> likely (> 4).
    assert wells(pe_most_likely=True, malignancy=True) == (4, "pe_unlikely")
    assert wells(pe_most_likely=True, prior_vte=True) == (4.5, "pe_likely")
    assert wells(dvt_signs=True, heart_rate=110) == (4.5, "pe_likely")


# --- Cockcroft-Gault (Cockcroft & Gault 1976) ---


def test_cockcroft_gault_formula() -> None:
    cg = REGISTRY["cockcroft_gault"]
    # Male, 40 y, 72 kg, SCr 1.0: (140-40)*72/(72*1) = 100 mL/min.
    score, _ = cg.evaluate({"age": 40, "weight": 72, "creatinine": 1.0, "sex": 0})
    assert score == pytest.approx(100.0)
    # Female, same: 85 mL/min.
    score, _ = cg.evaluate({"age": 40, "weight": 72, "creatinine": 1.0, "sex": 1})
    assert score == pytest.approx(85.0)
    # Male, 80 y, 60 kg, SCr 2.0: 60*60/144 = 25 mL/min.
    score, _ = cg.evaluate({"age": 80, "weight": 60, "creatinine": 2.0, "sex": 0})
    assert score == pytest.approx(25.0)


def test_cockcroft_gault_default_categories() -> None:
    cg = REGISTRY["cockcroft_gault"]
    assert cg.category(29.99) == "crcl_lt_30"
    assert cg.category(30) == "crcl_30_to_lt_60"
    assert cg.category(59.99) == "crcl_30_to_lt_60"
    assert cg.category(60) == "crcl_ge_60"


def test_cockcroft_gault_configurable_thresholds() -> None:
    cg = make_cockcroft_gault(thresholds=(15, 30, 60, 90))
    assert cg.category_names() == [
        "crcl_lt_15",
        "crcl_15_to_lt_30",
        "crcl_30_to_lt_60",
        "crcl_60_to_lt_90",
        "crcl_ge_90",
    ]
    assert get_calculator("cockcroft_gault", {"thresholds_ml_min": [50.0]}).category_names() == [
        "crcl_lt_50",
        "crcl_ge_50",
    ]


def test_registry_complete() -> None:
    assert set(REGISTRY) == {"heart", "curb65", "qsofa", "perc", "wells_pe", "cockcroft_gault"}
    for calc in REGISTRY.values():
        assert calc.citation
