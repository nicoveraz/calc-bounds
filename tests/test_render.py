import json

import pytest

from calc_bounds.calculators import REGISTRY
from calc_bounds.cohort import PatientCase, Trap, TrapKind, generate_cohort
from calc_bounds.config import CohortConfig
from calc_bounds.render import RenderedNote
from calc_bounds.render.export import export_review_sample
from calc_bounds.render.facts import build_facts, fmt
from calc_bounds.render.prompts import judge_prompt, render_prompt
from calc_bounds.render.validate import judge_issues, parse_judge, rule_issues
from calc_bounds.types import DocumentedState as D
from calc_bounds.units import to_canonical

CURB = REGISTRY["curb65"]

CASE = PatientCase(
    case_id="curb65-9999",
    seed=0,
    calculator="curb65",
    truth={
        "confusion": False,
        "urea": 7.3,
        "resp_rate": 31.0,
        "sbp": 118.0,
        "dbp": 70.0,
        "age": 72.0,
    },
    documented={
        "confusion": D.NEGATIVE,
        "urea": D.POSITIVE,
        "resp_rate": D.POSITIVE,
        "sbp": D.NEGATIVE,
        "dbp": D.NEGATIVE,
        "age": D.NOT_DOCUMENTED,
    },
    traps=[
        Trap(
            kind=TrapKind.CONTRADICTORY_VALUES, param="resp_rate", detail={"distractor_value": 24.0}
        )
    ],
    true_score=3,
    true_category="high",
    determined_from_note=False,
)

GOOD = (
    "CC: cough and fever for 4 days. HPI: productive cough, pleuritic pain. Alert and fully "
    "oriented. Exam: BP normal. RR 24 at triage, repeat RR 31/min. Crackles right base. "
    "Labs: urea 7.3 mmol/L, WBC 15. CXR right lower lobe consolidation. Plan: antibiotics, "
    "oxygen as needed, admit to medical ward for monitoring and IV therapy, reassess in the "
    "morning with repeat bloods and review of cultures and sensitivities."
)


def test_fmt() -> None:
    assert (fmt(112.0), fmt(7.3), fmt(1.25), fmt(80)) == ("112", "7.3", "1.25", "80")


def test_facts_instructions() -> None:
    facts = {f.param: f for f in build_facts(CASE, CURB)}
    assert facts["urea"].required_numbers == ["7.3"]
    assert "DO NOT MENTION" in facts["age"].instruction and facts["age"].leak_numbers == ["72"]
    assert facts["resp_rate"].required_numbers == ["31", "24"]
    assert "WITHOUT A NUMBER" in facts["sbp"].instruction
    prompt = render_prompt(CASE, CURB, "en-US")
    assert "[age] DO NOT MENTION" in prompt and "US English" in prompt
    assert "72" not in prompt.split("Fact sheet")[1].split("[age]")[1].split("\n")[0]


def test_comorbidity_trap_never_names_condition() -> None:
    case = generate_cohort(
        [REGISTRY["heart"]],
        CohortConfig(
            n_per_calculator=40,
            missingness=0.2,
            normal_as_negation_rate=0.5,
            undetermined_fraction=0.5,
            trap_rates={"comorbidity_via_medication": 1.0},
        ),
        seed=3,
    )
    with_trap = [c for c in case if c.traps]
    assert with_trap
    for c in with_trap:
        f = next(f for f in build_facts(c, REGISTRY["heart"]) if f.param == c.traps[0].param)
        assert f.instruction.startswith("STATE ONLY INDIRECTLY")
        assert "has:" not in f.instruction


def test_mixed_units_truth_matches_display() -> None:
    cases = generate_cohort(
        [REGISTRY["curb65"], REGISTRY["cockcroft_gault"]],
        CohortConfig(
            n_per_calculator=40,
            missingness=0.1,
            normal_as_negation_rate=0.0,
            undetermined_fraction=0.5,
            trap_rates={"mixed_units": 1.0},
        ),
        seed=5,
    )
    n = 0
    for c in cases:
        for t in c.traps:
            if t.kind == TrapKind.MIXED_UNITS:
                n += 1
                shown = float(t.detail["value_in_unit"])
                assert c.truth[t.param] == pytest.approx(
                    to_canonical(t.param, shown, str(t.detail["unit"]))
                )
                assert shown == round(shown)
                assert REGISTRY[c.calculator].evaluate(c.truth) == (c.true_score, c.true_category)
    assert n > 10


def test_rule_validation() -> None:
    assert rule_issues(CASE, CURB, GOOD) == []
    bad = (
        GOOD.replace("urea 7.3", "urea 7.30").replace("RR 24 at triage, ", "")
        + " Age 72. CURB-65 is 3."
    )
    problems = {
        (i.param, i.severity, i.problem.split(" ")[0]) for i in rule_issues(CASE, CURB, bad)
    }
    assert ("resp_rate", "error", "required") in problems  # distractor 24 missing
    assert ("age", "warning", "not-documented") in problems
    assert (None, "error", "names") in problems
    # 7.30 still contains the number 7.3 as written? No: exact surface form is required.
    assert ("urea", "error", "required") in problems
    assert any(i.problem.startswith("note too short") for i in rule_issues(CASE, CURB, "RR 31"))


def test_number_matching_boundaries() -> None:
    from calc_bounds.render.validate import _has_number

    assert _has_number("HR 112/min", "112")
    assert not _has_number("HR 1120", "112")
    assert not _has_number("urea 17.3", "7.3")
    assert not _has_number("value 7.35", "7.3")
    assert _has_number("RR 31.", "31")


def test_judge_parse_and_compare() -> None:
    verdicts = [
        ("confusion", "negated_or_normal", "Alert and fully oriented"),
        ("urea", "stated", "urea 7.3 mmol/L"),
        ("resp_rate", "stated", "repeat RR 31/min"),
        ("sbp", "negated_or_normal", "BP normal"),
        ("dbp", "negated_or_normal", "BP normal"),
        ("age", "implied", "elderly"),
    ]
    rows = [{"param": p, "status": s, "level": None, "quote": q} for p, s, q in verdicts]
    raw = "Here you go:\n" + json.dumps(rows)
    items = parse_judge(raw)
    assert items is not None and len(items) == 6
    issues = judge_issues(CASE, CURB, GOOD, items)
    assert [(i.param, i.severity) for i in issues] == [("age", "error"), ("age", "warning")]
    assert parse_judge("no json here") is None
    assert "urea" in judge_prompt(GOOD, CURB)


def test_export_review() -> None:
    note = RenderedNote(
        case_id=CASE.case_id,
        render="pilot",
        locale="en-US",
        text=GOOD,
        provider="session",
        model="m",
        cache_key="k",
    )
    md = export_review_sample([note], {CASE.case_id: CASE}, {"curb65": CURB}, {}, 0.2, seed=1)
    assert "curb65-9999" in md and "**not documented**" in md and "urea 7.3 mmol/L" in md


def test_judge_accepts_level_zero_as_stated_or_normal() -> None:
    heart = REGISTRY["heart"]
    truth = {p.id: False for p in heart.parameters} | {
        "heart_history": 0,
        "heart_ecg": 0,
        "age": 50.0,
        "heart_troponin": 0,
    }
    documented = {p.id: D.NOT_DOCUMENTED for p in heart.parameters} | {
        "heart_ecg": D.NEGATIVE,
        "heart_troponin": D.POSITIVE,
    }
    case = CASE.model_copy(update={"calculator": "heart", "truth": truth, "documented": documented})
    items = [
        {"param": p.id, "status": "not_mentioned", "level": None, "quote": None}
        for p in heart.parameters
    ]
    items[1] = {"param": "heart_ecg", "status": "stated", "level": "normal", "quote": None}
    items[-1] = {
        "param": "heart_troponin",
        "status": "negated_or_normal",
        "level": None,
        "quote": None,
    }
    assert judge_issues(case, heart, "", items) == []
    items[1]["level"] = "nonspecific_repolarization"
    assert [i.param for i in judge_issues(case, heart, "", items)] == ["heart_ecg"]


def test_not_mentioned_hints_in_prompt() -> None:
    perc = REGISTRY["perc"]
    documented = dict.fromkeys([p.id for p in perc.parameters], D.NOT_DOCUMENTED)
    truth = {p.id: False for p in perc.parameters} | {"age": 30.0, "heart_rate": 80.0, "spo2": 98.0}
    case = CASE.model_copy(update={"calculator": "perc", "truth": truth, "documented": documented})
    prompt = render_prompt(case, perc, "en-US")
    assert "Omit the medication list entirely" in prompt
    assert '"Medications: none"' in prompt
