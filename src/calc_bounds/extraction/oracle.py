"""Oracle extractor: reads ground truth (documented state + hidden truth) directly.

Upper bound for extraction; lets the pipeline run with zero API cost. It sees exactly what the
note documents: not-documented params are Unknown, never Absent. There is no note text, so
evidence spans are empty (note[0:0] == "" holds for any note).
"""

from collections.abc import Mapping

from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import ExtractionResult
from calc_bounds.types import (
    Absent,
    DocumentedState,
    EvidenceSpan,
    Extraction,
    NumericDomain,
    ParameterSpec,
    ParamId,
    Present,
    Unknown,
    Value,
)

_EMPTY = EvidenceSpan(start=0, end=0, text="")


def oracle_extractions(
    params: list[ParameterSpec] | tuple[ParameterSpec, ...],
    truth: Mapping[ParamId, Value],
    documented: Mapping[ParamId, DocumentedState],
) -> dict[ParamId, Extraction]:
    out: dict[ParamId, Extraction] = {}
    for p in params:
        match documented.get(p.id, DocumentedState.NOT_DOCUMENTED):
            case DocumentedState.POSITIVE:
                unit = p.domain.unit if isinstance(p.domain, NumericDomain) else None
                out[p.id] = Present(
                    value=truth[p.id],
                    unit=unit,
                    confidence=1.0,
                    confidence_source="oracle",
                    evidence=_EMPTY,
                )
            case DocumentedState.NEGATIVE:
                out[p.id] = Absent(confidence=1.0, confidence_source="oracle", evidence=_EMPTY)
            case DocumentedState.NOT_DOCUMENTED:
                out[p.id] = Unknown()
    return out


class OracleExtractor:
    name = "oracle"

    def __init__(self, cases: Mapping[str, PatientCase]) -> None:
        self.cases = cases

    def extract(self, case_id: str, note: str, params: list[ParameterSpec]) -> ExtractionResult:
        case = self.cases[case_id]
        return ExtractionResult(
            case_id=case_id, values=oracle_extractions(params, case.truth, case.documented)
        )
