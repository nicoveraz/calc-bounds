"""Cohorts and structured-truth mapping on SYNTHETIC fixtures (see make_fixtures.py)."""

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from calc_bounds.calculators import REGISTRY
from calc_bounds.mimic import cohorts as C
from calc_bounds.mimic import tables as T
from calc_bounds.mimic import truth as TR
from calc_bounds.mimic.criteria import IcdSet, load_criteria
from calc_bounds.units import to_canonical

FIX = Path(__file__).parent / "fixtures" / "mimic_synthetic"
DIRS = T.TableDirs(
    hosp=FIX / "hosp", icu=FIX / "icu", ed=FIX / "ed", notes_path=FIX / "note" / "discharge.csv"
)
CRIT = load_criteria(Path("configs/mimic_criteria.yaml"))
ALL_SUBJECTS = set(range(99000001, 99000007))


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return T.load_edstays(DIRS)


@pytest.fixture(scope="module")
def records(stays: pd.DataFrame) -> dict[int, TR.StayRecord]:
    ids = CRIT.itemids
    labs = T.load_labevents(DIRS, {ids.bun, ids.creatinine, ids.troponin_t}, ALL_SUBJECTS)
    gcs = T.load_chartevents(DIRS, {ids.gcs_eye, ids.gcs_verbal, ids.gcs_motor}, ALL_SUBJECTS)
    omr = T.load_omr(DIRS, {CRIT.omr.weight_lbs, CRIT.omr.bmi}, ALL_SUBJECTS)
    return TR.assemble(
        [
            TR.vitals(stays, T.load_triage(DIRS), T.load_vitalsign(DIRS, stays["stay_id"]), CRIT),
            TR.demographics(stays, T.load_patients(DIRS)),
            TR.labs(stays, labs, CRIT),
            TR.gcs(stays, gcs, CRIT),
            TR.omr_values(stays, omr, CRIT),
            TR.comorbidities(stays, T.load_diagnoses_icd(DIRS), CRIT),
        ]
    )


# --- cohorts ------------------------------------------------------------------------------------


def test_pneumonia_principal_diagnosis_only(stays: pd.DataFrame) -> None:
    dx = T.load_diagnoses_icd(DIRS)
    assert C.pneumonia_stays(stays, dx, CRIT.cohorts) == {97000001, 97000004}  # ICD-10 and -9
    any_position = CRIT.cohorts.model_copy(update={"pneumonia_max_seq_num": None})
    assert C.pneumonia_stays(stays, dx, any_position) == {97000001, 97000004, 97000005}


def test_chest_pain_by_complaint_or_ed_diagnosis(stays: pd.DataFrame) -> None:
    got = C.chest_pain_stays(stays, T.load_triage(DIRS), T.load_ed_diagnosis(DIRS), CRIT.cohorts)
    assert got == {97000002, 97000003, 97000006}  # "Chest pain", ED dx R079, "CHEST PRESSURE"


def test_suspected_infection_needs_systemic_antibiotic_and_culture(stays: pd.DataFrame) -> None:
    rx = T.load_prescriptions(DIRS, {98000001, 98000002})
    micro = T.load_microbiologyevents(DIRS, ALL_SUBJECTS)
    # 99000002 has a culture but only eye drops (route OU): excluded.
    assert C.suspected_infection_stays(stays, rx, micro, CRIT) == {97000001}


def test_suspicion_pairing_windows() -> None:
    s = CRIT.cohorts.suspected_infection
    t0 = pd.Timestamp("2150-01-01 08:00")
    stays = pd.DataFrame({"stay_id": [1], "subject_id": [1], "intime": [t0]})

    def times(abx_h: float, cult_h: float) -> list:
        abx = pd.DataFrame({"subject_id": [1], "time": [t0 + pd.Timedelta(hours=abx_h)]})
        cult = pd.DataFrame({"subject_id": [1], "time": [t0 + pd.Timedelta(hours=cult_h)]})
        return list(C.suspicion_times(stays, abx, cult, s, (-6, 24))["suspicion_time"])

    assert times(abx_h=70, cult_h=0) == [t0]  # culture, then antibiotic within 72 h
    assert times(abx_h=80, cult_h=0) == []  # too late
    assert times(abx_h=1, cult_h=20) == [t0 + pd.Timedelta(hours=1)]  # antibiotic first, 19 h
    assert times(abx_h=1, cult_h=30) == []  # culture more than 24 h after the antibiotic
    assert times(abx_h=40, cult_h=30) == []  # paired, but suspicion time outside the window


def test_lab_window_cohort(stays: pd.DataFrame) -> None:
    labs = T.load_labevents(DIRS, {CRIT.itemids.bun}, ALL_SUBJECTS)
    # 99000004's only BUN is 48 h before arrival.
    assert C.lab_in_window_stays(stays, labs, CRIT.itemids.bun, 24) == {97000001}


# --- structured truth ---------------------------------------------------------------------------


def test_first_ed_values(records: dict[int, TR.StayRecord]) -> None:
    r = records[97000001].truth
    assert r["resp_rate"] == 32 and r["sbp"] == 85 and r["dbp"] == 50
    assert r["age"] == 70.0 and r["sex"] == 0
    assert r["urea"] == pytest.approx(to_canonical("urea", 30, "bun_mg/dL"))  # first BUN only
    assert r["creatinine"] == 1.5
    assert r["weight"] == pytest.approx(154 * 0.45359237)  # nearest omr weight, not 200 lb
    assert r["altered_mentation"] is False and r["confusion"] is False  # GCS 4+5+6
    assert records[97000001].source["resp_rate"] == "triage"


def test_vitalsign_fallback_and_implausible(records: dict[int, TR.StayRecord]) -> None:
    r4 = records[97000004]
    assert r4.truth["resp_rate"] == 26 and r4.source["resp_rate"] == "vitalsign"
    assert "urea" not in r4.truth  # BUN outside the lab window
    r6 = records[97000006]
    assert r6.truth["heart_rate"] == 76  # triage 999 dropped; first vitalsign in window
    assert "heart_rate" in r6.implausible


def test_troponin_levels(records: dict[int, TR.StayRecord]) -> None:
    assert records[97000002].truth["heart_troponin"] == 1  # 0.02 / ULN 0.01 = 2x
    assert records[97000006].truth["heart_troponin"] == 0  # "<0.01", ULN 0.01
    assert TR.troponin_level(0.01, None, 0.01) == 0
    assert TR.troponin_level(0.03, None, 0.01) == 1
    assert TR.troponin_level(0.031, None, 0.01) == 2
    assert TR.troponin_level(float("nan"), "<0.05", 0.01) is None  # censored, not settled
    assert TR.troponin_level(float("nan"), ">0.5", 0.01) == 2
    assert TR.troponin_level(0.5, None, float("nan")) is None
    # result only in the comments column (value empty)
    assert TR.troponin_level(float("nan"), "", 0.01, "LESS THAN 0.01") == 0
    assert TR.troponin_level(float("nan"), None, 0.01, "greater than 0.5 ng/mL") == 2
    assert TR.troponin_level(float("nan"), None, 0.01, "___") is None


def test_comorbidities_true_or_unknown_never_false(records: dict[int, TR.StayRecord]) -> None:
    r2 = records[97000002].truth
    assert r2["hypertension"] is True and r2["hypercholesterolemia"] is True  # I10, ICD-9 2724
    for p in ("diabetes", "smoking", "family_history_cad", "atherosclerotic_disease"):
        assert p not in r2  # no code: Unknown, not False
    assert records[97000006].truth["obesity"] is True  # BMI 32.1 (omr)
    assert records[97000006].source["obesity"] == "omr:bmi"


def test_gcs_absent_outside_icu(records: dict[int, TR.StayRecord]) -> None:
    for stay in (97000002, 97000004, 97000006):
        assert "altered_mentation" not in records[stay].truth


def test_make_case_and_annotations(records: dict[int, TR.StayRecord]) -> None:
    heart = REGISTRY["heart"]
    case = TR.make_case(
        heart, records[97000002], subject_id=99000002, hadm_id=98000002, note_id="n", crit=CRIT
    )
    assert case.case_id == "heart-97000002"
    assert set(case.truth) <= {p.id for p in heart.parameters}
    assert case.needs_annotation == ["heart_history", "heart_ecg"]
    assert TR.reference_category(heart, case) is None  # judgement items unknown
    done = TR.apply_annotations(case, {"heart_history": 2, "heart_ecg": 0})
    assert done.needs_annotation == [] and done.truth_source["heart_history"] == "annotation"
    assert case.truth.get("heart_history") is None  # original unchanged


def test_reference_category_when_structured_truth_settles_it(
    records: dict[int, TR.StayRecord],
) -> None:
    curb = REGISTRY["curb65"]
    case = TR.make_case(
        curb, records[97000001], subject_id=99000001, hadm_id=98000001, note_id="n", crit=CRIT
    )
    assert set(case.truth) == {p.id for p in curb.parameters}
    assert TR.reference_category(curb, case) == "high"  # urea, RR, BP, age: 4 points


codes = st.lists(
    st.tuples(
        st.sampled_from(["9", "10"]),
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=7),
    ),
    max_size=30,
)


@given(codes)
def test_icd_mapping_never_produces_false(dx_codes: list[tuple[str, str]]) -> None:
    """Unknown is never Absent: comorbidity values from codes are only ever True."""
    stays = pd.DataFrame({"stay_id": [1], "hadm_id": [10]})
    dx = pd.DataFrame(
        {
            "hadm_id": [10] * len(dx_codes),
            "icd_version": [int(v) for v, _ in dx_codes],
            "icd_code": [c for _, c in dx_codes],
        }
    )
    long = TR.comorbidities(stays, dx, CRIT)
    assert set(long["value"]) <= {True}
    records = TR.assemble([long])
    for rec in records.values():
        assert all(v is True for v in rec.truth.values())


def test_icd_prefix_matching() -> None:
    s = IcdSet(icd9=["486"], icd10=["J18"])
    df = pd.DataFrame(
        {"icd_code": ["J189", "486", "J18", "4860", "J12"], "icd_version": [10, 9, 9, 10, 10]}
    )
    assert list(s.matches(df)) == [True, True, False, False, False]  # version must match
    with pytest.raises(ValueError):
        IcdSet(icd10=["J18.9"])
