"""Extractors return tri-state `Extraction`s per parameter.

Implementations: oracle (reads ground truth; zero cost), LLM (structured output; logprob
confidence where available, else self-reported and flagged). Every evidence span is checked to
be an exact substring of the note; mismatches are rejected and logged.
"""

from typing import Protocol

from pydantic import BaseModel

from calc_bounds.llm import Usage
from calc_bounds.types import EvidenceSpan, Extraction, ParameterSpec, ParamId


class RejectedSpan(BaseModel):
    param: ParamId
    span: EvidenceSpan | None
    reason: str


class ExtractionResult(BaseModel):
    case_id: str
    values: dict[ParamId, Extraction]
    rejected: list[RejectedSpan] = []
    usage: Usage | None = None
    extractor: str = ""
    render: str = ""


class Extractor(Protocol):
    name: str

    def extract(self, case_id: str, note: str, params: list[ParameterSpec]) -> ExtractionResult: ...
