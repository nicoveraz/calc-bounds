"""EHR-as-clinician and physician annotation (template export / import), synthetic cases."""

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from calc_bounds.mimic.annotation import annotation_template, parse_value, read_annotations
from calc_bounds.mimic.clinician import EHRClinician
from calc_bounds.mimic.truth import SPECS, MimicCase, apply_annotations
from calc_bounds.simulator import SimulatedClinician


def _case(i: int, calculator: str = "heart") -> MimicCase:
    return MimicCase(
        case_id=f"{calculator}-{97000000 + i}",
        calculator=calculator,
        subject_id=99000000 + i,
        stay_id=97000000 + i,
        hadm_id=98000000 + i,
        note_id=f"{98000000 + i}-DS-1",
        truth={"age": 50.0, "hypertension": True},
        truth_source={"age": "patients", "hypertension": "diagnoses_icd"},
        needs_annotation=["heart_history", "heart_ecg"] if calculator == "heart" else [],
    )


def test_ehr_clinician_answers_from_record_or_not_available() -> None:
    c = EHRClinician({"age": 70.0, "confusion": False})
    assert isinstance(c, SimulatedClinician)
    assert c.ask("age").value == 70.0
    a = c.ask("confusion")
    assert a.status == "answered" and a.value is False  # recorded negative stays negative
    assert c.ask("urea").status == "not_available"  # missing from the record
    assert [a.param for a in c.log] == ["age", "confusion", "urea"]


record = st.dictionaries(
    st.sampled_from(sorted(SPECS)),
    st.one_of(st.booleans(), st.integers(0, 2), st.floats(0, 200, allow_nan=False)),
    max_size=10,
)


@given(record, st.lists(st.sampled_from(sorted(SPECS)), max_size=20))
def test_ehr_clinician_never_invents_values(rec: dict, questions: list[str]) -> None:
    c = EHRClinician(rec)
    for q in questions:
        a = c.ask(q)
        if q in rec:
            assert a.status == "answered" and a.value == rec[q]
        else:
            assert a.status == "not_available" and a.value is None


def test_template_has_ids_only_and_is_seeded() -> None:
    cases = [_case(i) for i in range(1, 8)] + [_case(9, "curb65")]
    t = annotation_template(cases, None, seed=1)
    assert len(t) == 14 and set(t["param"]) == {"heart_history", "heart_ecg"}
    assert (t["value"] == "").all() and "text" not in t.columns
    a = annotation_template(cases, 3, seed=1)
    b = annotation_template(cases, 3, seed=1)
    assert a.equals(b) and a["case_id"].nunique() == 3


def test_parse_values() -> None:
    hist, ecg = SPECS["heart_history"], SPECS["heart_ecg"]
    assert parse_value(hist, "highly_suspicious") == 2
    assert parse_value(ecg, " Normal ") == 0
    assert parse_value(SPECS["pe_most_likely"], "yes") is True
    assert parse_value(SPECS["pe_most_likely"], "no") is False
    assert parse_value(hist, "") is None and parse_value(hist, "unknown") is None
    with pytest.raises(ValueError):
        parse_value(hist, "very")


def test_import_roundtrip(tmp_path: Path) -> None:
    cases = [_case(1), _case(2)]
    t = annotation_template(cases, None, seed=1).set_index(["case_id", "param"])
    a, b = cases[0].case_id, cases[1].case_id
    t.loc[(a, "heart_history"), "value"] = "moderately_suspicious"
    t.loc[(a, "heart_ecg"), "value"] = "unknown"
    t.loc[(b, "heart_ecg"), "value"] = "significant_st_deviation"
    path = tmp_path / "filled.csv"
    t.reset_index().to_csv(path, index=False)
    got = read_annotations(path, cases)
    assert got == {a: {"heart_history": 1}, b: {"heart_ecg": 2}}  # "unknown" stays Unknown
    done = apply_annotations(cases[0], got[a])
    assert done.truth["heart_history"] == 1 and done.needs_annotation == ["heart_ecg"]


def test_import_reports_every_error(tmp_path: Path) -> None:
    cases = [_case(1)]
    rows = annotation_template(cases, None, seed=1)
    bad = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    bad.loc[0, "value"] = "maybe"
    bad.loc[2, "case_id"] = "heart-1"
    extra = bad.iloc[[1]].assign(param="age")
    path = tmp_path / "bad.csv"
    pd.concat([bad, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError) as e:
        read_annotations(path, cases)
    msg = str(e.value)
    assert "maybe" in msg and "unknown case" in msg and "not pending" in msg
