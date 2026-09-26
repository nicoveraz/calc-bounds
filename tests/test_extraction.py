import json
import math

import pytest

from calc_bounds.bounds import Exact, Interval, from_extractions
from calc_bounds.calculators import REGISTRY
from calc_bounds.extraction.confidence import field_confidence
from calc_bounds.extraction.llm import extraction_prompt, extraction_schema, parse_extraction
from calc_bounds.types import Absent, Present, Unknown

CURB = REGISTRY["curb65"]
NOTE = (
    "72M with cough. Alert and oriented, not confused. RR 24 at triage, repeat RR 31. "
    "BP normal. BUN 23 mg/dL."
)


def item(status: str, value=None, unit=None, evidence=None, confidence=0.9) -> dict:
    return {
        "status": status,
        "value": value,
        "unit": unit,
        "evidence": evidence,
        "confidence": confidence,
    }


def output(**items: dict) -> str:
    base = {p.id: item("unknown") for p in CURB.parameters}
    return json.dumps(base | items)


def test_prompt_and_schema() -> None:
    prompt = extraction_prompt(NOTE, CURB)
    assert NOTE in prompt and "bun_mg/dL" in prompt and "never" in prompt
    schema = extraction_schema(CURB)
    assert schema["required"] == [p.id for p in CURB.parameters]
    assert schema["properties"]["urea"]["properties"]["unit"]["anyOf"][0]["enum"][0] == "mmol/L"


def test_parse_tristate_units_and_spans() -> None:
    text = output(
        confusion=item("absent", evidence="not confused"),
        urea=item("present", 23, "bun_mg/dL", "BUN 23 mg/dL"),
        resp_rate=item("present", 31, "/min", "repeat RR 31"),
        sbp=item("absent", evidence="BP normal"),
    )
    r = parse_extraction("c1", NOTE, CURB, text)
    assert r.rejected == []
    assert isinstance(r.values["confusion"], Absent)
    assert isinstance(r.values["age"], Unknown) and isinstance(r.values["dbp"], Unknown)
    urea = r.values["urea"]
    assert isinstance(urea, Present) and urea.unit == "bun_mg/dL"
    assert NOTE[urea.evidence.start : urea.evidence.end] == "BUN 23 mg/dL"
    assert urea.confidence_source == "self_reported"
    known = from_extractions(CURB, r.values)
    assert isinstance(known["urea"], Exact)
    assert known["urea"].value == pytest.approx(23 / 2.8014)  # code converts, not the model
    assert known["sbp"] == Interval(lo=101, hi=139)
    assert "age" not in known  # Unknown is never Absent


def test_rejections_become_unknown_and_are_logged() -> None:
    text = output(
        urea=item("present", 23, "bun_mg/dL", "BUN 23mg/dl"),  # not an exact substring
        age=item("absent", evidence="72M"),  # age cannot be absent
        resp_rate=item("present", 400, "/min", "repeat RR 31"),  # outside plausible range
        confusion=item("present", evidence="not confused", value="yes"),
    )
    r = parse_extraction("c1", NOTE, CURB, text)
    reasons = {x.param: x.reason for x in r.rejected}
    assert set(reasons) == {"urea", "age", "resp_rate"}
    assert "exact substring" in reasons["urea"]
    for pid in reasons:
        assert isinstance(r.values[pid], Unknown)
    assert isinstance(r.values["confusion"], Present)  # bool value is ignored: present = True


def test_unparseable_output() -> None:
    r = parse_extraction("c1", NOTE, CURB, "sorry, I cannot")
    assert all(isinstance(v, Unknown) for v in r.values.values())
    assert r.rejected and r.rejected[0].param == "*"


def test_logprob_confidence() -> None:
    text = '{"a": {"status": "present", "value": 31}, "b": {"status": "unknown", "value": null}}'
    # Tokenize per character with logprob 0 except the status of "a" (two tokens) and value.
    toks = [{"token": ch, "logprob": 0.0} for ch in text]
    a_status = text.index('"present"')
    toks[a_status + 1]["logprob"] = math.log(0.8)
    a_value = text.index("31")
    toks[a_value]["logprob"] = math.log(0.5)
    assert field_confidence(text, toks, "a") == pytest.approx(0.4)
    assert field_confidence(text, toks, "b") == pytest.approx(1.0)
    assert field_confidence(text, toks, "zzz") is None
    assert field_confidence(text + " ", toks, "a") is None  # tokens must reproduce the text


def test_claim_correct_level_zero_equivalence() -> None:
    from calc_bounds.cohort import PatientCase
    from calc_bounds.eval.metrics import claim_correct
    from calc_bounds.types import DocumentedState as D
    from calc_bounds.types import EvidenceSpan

    heart = REGISTRY["heart"]
    truth = {p.id: False for p in heart.parameters} | {
        "heart_history": 1,
        "heart_ecg": 0,
        "age": 50.0,
        "heart_troponin": 2,
    }
    documented = dict.fromkeys(truth, D.NOT_DOCUMENTED) | {
        "heart_ecg": D.NEGATIVE,
        "heart_troponin": D.POSITIVE,
    }
    case = PatientCase(
        case_id="h",
        seed=0,
        calculator="heart",
        truth=truth,
        documented=documented,
        true_score=0,
        true_category="low",
        determined_from_note=False,
    )
    ev = EvidenceSpan(start=0, end=1, text="x")
    kw = {"confidence": 1.0, "confidence_source": "self_reported", "evidence": ev}
    assert claim_correct(case, "heart_ecg", Present(value=0, **kw))
    assert claim_correct(case, "heart_ecg", Absent(**kw))
    assert not claim_correct(case, "heart_ecg", Present(value=1, **kw))
    assert claim_correct(case, "heart_troponin", Present(value=2, **kw))
    assert not claim_correct(case, "heart_troponin", Absent(**kw))  # level 2 is not normal
    assert not claim_correct(case, "heart_history", Present(value=1, **kw))  # not documented


def test_renderer_bias_interaction() -> None:
    import pandas as pd

    from calc_bounds.eval.bias import interaction, renderer_bias

    rows = []
    # Extractor A: 1.0 on A-notes, 0.8 on B-notes; extractor B: 0.9 on both -> interaction 0.2
    for cid in range(10):
        for x, r, ok in [("A", "a", 10), ("A", "b", 8), ("B", "a", 9), ("B", "b", 9)]:
            for k in range(10):
                rows.append(
                    {"extractor": x, "render": r, "case_id": cid, "param": k, "correct": k < ok}
                )
    df = pd.DataFrame(rows)
    assert interaction(df, ("A", "B"), ("a", "b")) == pytest.approx(0.2)
    rep = renderer_bias(df, ("A", "B"), ("a", "b"), n_boot=200)
    assert rep["n_cases"] == 10 and rep["ci95"][0] == pytest.approx(0.2)
