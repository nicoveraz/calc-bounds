"""MIMIC outcomes: claim scoring, note-alone bounds, policies with the EHR as clinician.

Property tests (hypothesis) on random calculators, records and notes: if the note only states
values that are in the record, the reference category always lies within the note's bounds,
and S3 with the EHR as clinician never commits prematurely or to a wrong category.
"""

import math

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import partial_and_completion, value_of

from calc_bounds.calculators import REGISTRY, Calculator
from calc_bounds.extraction import ExtractionResult
from calc_bounds.extraction.oracle import PrecomputedExtractor, oracle_extractions
from calc_bounds.mimic.clinician import EHRClinician
from calc_bounds.mimic.outcomes import (
    aggregate_tables,
    case_row,
    claim_agrees,
    claim_rows,
    policy_row,
    proportions,
    suppress_counts,
    wilson,
)
from calc_bounds.mimic.truth import MimicCase, reference_category
from calc_bounds.policies import BoundsPolicy, Trace
from calc_bounds.types import (
    Absent,
    BoolDomain,
    DocumentedState,
    EvidenceSpan,
    NumericDomain,
    Present,
    Unknown,
)

CALCS = st.sampled_from(sorted(REGISTRY))
SPAN = EvidenceSpan(start=0, end=0, text="")


def _present(v: object, unit: str | None = None) -> Present:
    return Present(
        value=v,  # type: ignore[arg-type]
        unit=unit,
        confidence=1,
        confidence_source="oracle",
        evidence=SPAN,
    )


def _absent() -> Absent:
    return Absent(confidence=1, confidence_source="oracle", evidence=SPAN)


def _run(policy: BoundsPolicy, case: MimicCase, calc: Calculator, ext: ExtractionResult) -> Trace:
    extractor = PrecomputedExtractor("t", {case.case_id: ext})
    return policy.run(case, "", calc, extractor, EHRClinician(case.truth))


def _case(calc: Calculator, truth: dict) -> MimicCase:
    return MimicCase(
        case_id=f"{calc.id}-1",
        calculator=calc.id,
        subject_id=1,
        stay_id=1,
        hadm_id=1,
        note_id="n",
        truth=truth,
        truth_source=dict.fromkeys(truth, "test"),
        needs_annotation=[],
    )


@st.composite
def record_and_note(draw: st.DrawFn) -> tuple[Calculator, MimicCase, ExtractionResult]:
    """A record (subset of a full truth) and a note that states a subset of the record,
    correctly (values exactly; normal values possibly as 'normal' / negated)."""
    calc = REGISTRY[draw(CALCS)]
    full = {p.id: draw(value_of(calc, p)) for p in calc.parameters}
    record = {p: v for p, v in full.items() if draw(st.booleans())}
    documented: dict[str, DocumentedState] = {}
    for p, v in record.items():
        if not draw(st.booleans()):
            continue
        spec = calc.param(p)
        normal = (
            (isinstance(spec.domain, BoolDomain) and v is False)
            or (
                isinstance(spec.domain, NumericDomain)
                and spec.absent_means is not None
                and spec.absent_means[0] <= v <= spec.absent_means[1]
            )
        ) and spec.negatable
        negate = normal and (isinstance(spec.domain, BoolDomain) or draw(st.booleans()))
        documented[p] = DocumentedState.NEGATIVE if negate else DocumentedState.POSITIVE
    values = oracle_extractions(calc.parameters, record, documented)
    return calc, _case(calc, record), ExtractionResult(case_id=f"{calc.id}-1", values=values)


@settings(max_examples=150, deadline=None)
@given(record_and_note())
def test_reference_within_note_bounds(x: tuple[Calculator, MimicCase, ExtractionResult]) -> None:
    calc, case, ext = x
    row = case_row(case, calc, ext, fallback_full_text=False)
    if row["reference_determined"]:
        assert row["truth_within_note_bounds"] is True
    for r in claim_rows(case, calc, ext):
        assert r["agrees"] in (True, None)  # a faithful note never disagrees with the record


@settings(max_examples=100, deadline=None)
@given(record_and_note())
def test_s3_with_ehr_clinician_is_sound(x: tuple[Calculator, MimicCase, ExtractionResult]) -> None:
    calc, case, ext = x
    extractor = PrecomputedExtractor("t", {case.case_id: ext})
    t = BoundsPolicy().run(case, "", calc, extractor, EHRClinician(case.truth))
    row = policy_row(case, calc, t)
    ref = reference_category(calc, case)
    assert not t.committed_while_undetermined
    if ref is not None:
        assert row["truth_within_final_bounds"] is True
        assert t.final_category in (None, ref)
        # The record settles the category, so S3 can always commit using the EHR's answers.
        assert t.final_category == ref
    for s in t.steps:  # every question was decision-relevant when asked
        assert s.relevant is not None and s.question in s.relevant


@settings(max_examples=200, deadline=None)
@given(
    CALCS.flatmap(lambda c: st.tuples(st.just(c), partial_and_completion(REGISTRY[c]))),
    st.data(),
)
def test_present_agreement_means_same_points(x: tuple, data: st.DataObject) -> None:
    """For step params, an 'agreeing' note value scores exactly like the recorded value."""
    calc_id, (_, completion) = x
    calc = REGISTRY[calc_id]
    for p in calc.parameters:
        if not isinstance(p.domain, NumericDomain) or p.id in calc.continuous:
            continue
        note_value = data.draw(value_of(calc, p))
        agrees, _ = claim_agrees(calc, p, _present(note_value, p.domain.unit), completion[p.id])
        same = calc.score({**completion, p.id: note_value}) == calc.score(completion)
        if agrees:
            assert same


def test_claim_agreement_examples() -> None:
    curb, cg = REGISTRY["curb65"], REGISTRY["cockcroft_gault"]
    rr, urea, age = curb.param("resp_rate"), curb.param("urea"), cg.param("age")
    assert claim_agrees(curb, rr, _present(31, "/min"), 34.0) == (True, False)  # same points
    assert claim_agrees(curb, rr, _present(29, "/min"), 30.0) == (False, False)
    assert claim_agrees(curb, rr, _present(30, "/min"), 30.0) == (True, True)
    assert claim_agrees(curb, urea, _present(28, "bun_mg/dL"), 10.0)[0] is True  # 10 mmol/L
    assert claim_agrees(curb, rr, _absent(), 16.0) == (True, None)  # "RR normal", 16 recorded
    assert claim_agrees(curb, rr, _absent(), 24.0) == (False, None)
    assert claim_agrees(curb, curb.param("confusion"), _absent(), False) == (True, None)
    assert claim_agrees(curb, rr, Unknown(), 24.0) == (None, None)
    assert claim_agrees(curb, rr, _present(24, "/min"), None) == (None, None)  # not in record
    assert claim_agrees(cg, age, _present(70, "years"), 72.0) == (True, False)  # within 5%
    assert claim_agrees(cg, age, _present(60, "years"), 72.0) == (False, False)


def test_missing_as_normal_under_triage() -> None:
    """Note silent on confusion; record says confused. Missing = normal under-triages."""
    curb = REGISTRY["curb65"]
    truth = {"confusion": True, "urea": 8.0, "resp_rate": 20.0, "sbp": 120.0, "dbp": 70.0}
    truth |= {"age": 50.0}
    case = _case(curb, truth)
    note = {p: DocumentedState.POSITIVE for p in truth if p != "confusion"}
    values = oracle_extractions(curb.parameters, truth, note)
    ext = ExtractionResult(case_id=case.case_id, values=values)
    row = case_row(case, curb, ext, fallback_full_text=False)
    assert row["reference_category"] == "moderate" and row["determined_by_note"] is False
    assert row["missing_as_normal_category"] == "low"
    assert row["missing_as_normal_under_triage"] is True
    t = _run(BoundsPolicy(binary=True), case, curb, ext)
    r = policy_row(case, curb, t)
    assert r["final_category"] == "low" and r["under_triage"] and not r["truth_within_final_bounds"]
    s3 = _run(BoundsPolicy(), case, curb, ext)
    assert s3.final_category == "moderate" and s3.n_questions == 1


def test_wilson_and_proportions() -> None:
    lo, hi = wilson(3, 10)
    assert 0.10 < lo < 0.11 and 0.60 < hi < 0.61
    assert all(math.isnan(v) for v in wilson(0, 0))
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "m": [True, False, None, True]})
    p = proportions(df, ["g"], ["m"]).set_index("g")
    assert p.loc["a", "k"] == 1 and p.loc["a", "n"] == 2  # NA outside the denominator
    assert p.loc["b", "rate"] == 1.0


def test_aggregates_suppress_small_groups_and_carry_no_ids() -> None:
    curb = REGISTRY["curb65"]
    truth = {"confusion": False, "urea": 5.0, "resp_rate": 20.0, "sbp": 120.0, "dbp": 70.0}
    truth |= {"age": 50.0}
    cases, rows, pol, claims = [], [], [], []
    for i in range(12):
        case = _case(curb, truth).model_copy(update={"case_id": f"curb65-{i}"})
        ext = ExtractionResult(case_id=case.case_id, values={})
        cases.append(case)
        rows.append(case_row(case, curb, ext, False))
        claims += claim_rows(case, curb, ext)
        t = _run(BoundsPolicy(), case, curb, ext)
        pol.append(policy_row(case, curb, t))
    tables = aggregate_tables(pd.DataFrame(rows), pd.DataFrame(pol), pd.DataFrame(claims), 10)
    note = tables["note_alone"].set_index(["calculator", "metric"])
    assert note.loc[("curb65", "determined_by_note"), "n"] == 12
    assert note.loc[("curb65", "determined_by_note"), "rate"] == 0.0
    q = tables["questions"].set_index(["calculator", "policy"])
    assert q.loc[("curb65", "s3_bounds"), "mean_questions"] > 0
    few = [pd.DataFrame(rows[:3]), pd.DataFrame(pol[:3]), pd.DataFrame(claims[:18])]
    small = aggregate_tables(*few, 10)
    assert small["note_alone"]["suppressed"].all() and small["note_alone"]["rate"].isna().all()
    for t in tables.values():
        text = t.to_csv(index=False)
        assert "curb65-1" not in text and "case_id" not in text
    flow = pd.DataFrame({"calculator": ["x"], "eligible": [500], "admitted": [3], "with_note": [0]})
    s = suppress_counts(flow, ["eligible", "admitted", "with_note"], 10)
    assert list(s.iloc[0]) == ["x", 500, "<10", 0]
