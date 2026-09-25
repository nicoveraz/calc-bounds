"""LLM renderer: structured patient -> clinical note, per locale, directly from the structure.

Never translate one rendered note into another locale. Style guides live in render/styles/.
"""

from typing import Literal

from pydantic import BaseModel

from calc_bounds.types import ParamId


class ValidationIssue(BaseModel):
    param: ParamId | None
    source: Literal["rule", "judge"]
    severity: Literal["error", "warning"]
    problem: str


class RenderedNote(BaseModel):
    case_id: str
    render: str
    """Name of the render set (config `renders` key)."""
    locale: str
    text: str
    provider: str
    model: str
    cache_key: str
    issues: list[ValidationIssue] = []
    """Deterministic (rule) validation. Judge results are stored separately. Failures are
    flagged, never silently dropped."""


class JudgeResult(BaseModel):
    case_id: str
    render: str
    locale: str
    judge_model: str
    parsed: bool
    issues: list[ValidationIssue]
    raw: str
