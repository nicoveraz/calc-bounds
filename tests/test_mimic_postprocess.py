"""Post-extraction rules on MIMIC cases (synthetic values only)."""

from calc_bounds.calculators import get_calculator
from calc_bounds.extraction import ExtractionResult
from calc_bounds.mimic.extract import bun_unit_rule, with_structured_params
from calc_bounds.mimic.truth import MimicCase
from calc_bounds.types import EvidenceSpan, Present, Unknown
from calc_bounds.units import to_canonical


def _urea(text: str, unit: str | None) -> ExtractionResult:
    ev = EvidenceSpan(start=0, end=len(text), text=text)
    e = Present(value=28.0, unit=unit, confidence=0.9, confidence_source="logprob", evidence=ev)
    return ExtractionResult(case_id="curb65-97000001", values={"urea": e})


def test_bun_quote_is_read_as_bun_mg_dl() -> None:
    for unit in (None, "mmol/L", "mg/dL"):
        e = bun_unit_rule(_urea("BUN-28", unit)).values["urea"]
        assert e.unit == "bun_mg/dL"
        assert abs(to_canonical("urea", e.value, e.unit) - 10.0) < 0.05  # 28 mg/dL BUN
    for quote in ("BLOOD UreaN-28", "urea nitrogen 28", "Urea N 28"):
        assert bun_unit_rule(_urea(quote, "mmol/L")).values["urea"].unit == "bun_mg/dL"
    assert bun_unit_rule(_urea("Urea 28 mmol/L", "mmol/L")).values["urea"].unit == "mmol/L"
    assert bun_unit_rule(_urea("BUN 28", "urea_mg/dL")).values["urea"].unit == "urea_mg/dL"


def test_age_and_sex_come_from_the_record_with_valid_evidence() -> None:
    calc = get_calculator("cockcroft_gault")
    case = MimicCase(
        case_id="cockcroft_gault-97000001",
        calculator="cockcroft_gault",
        subject_id=99000001,
        stay_id=97000001,
        hadm_id=98000001,
        note_id="98000001-DS-1",
        truth={"age": 72.0, "sex": 1, "creatinine": 1.1},
        truth_source={"age": "patients", "sex": "patients", "creatinine": "labevents"},
        needs_annotation=[],
    )
    note = "SYNTHETIC. ___ y/o with creatinine 1.1."
    ext = ExtractionResult(case_id=case.case_id, values={"age": Unknown(), "sex": Unknown()})
    out, text = with_structured_params(calc, case, ext, note, ("age", "sex"))
    assert text.startswith(note)  # original offsets unchanged
    for pid, expected in (("age", 72.0), ("sex", 1)):
        e = out.values[pid]
        assert isinstance(e, Present) and e.value == expected
        assert text[e.evidence.start : e.evidence.end] == e.evidence.text
    assert "female" in out.values["sex"].evidence.text
