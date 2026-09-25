"""LLM renderer: structured patient -> clinical note, per locale, directly from the structure.

Never translate one rendered note into another locale. Style guides live in render/styles/.
"""

from pydantic import BaseModel

from calc_bounds.cohort import PatientCase
from calc_bounds.types import ParamId


class ValidationIssue(BaseModel):
    param: ParamId
    problem: str
    """e.g. 'documented value missing', 'negation not expressed', 'not-documented inferable'."""


class RenderedNote(BaseModel):
    case_id: str
    locale: str
    text: str
    provider: str
    model: str
    issues: list[ValidationIssue] = []
    """Filled by the validation pass. Failures are flagged, never silently dropped."""


def render_note(case: PatientCase, locale: str, style_guide: str) -> RenderedNote:
    raise NotImplementedError  # M3


def validate_note(case: PatientCase, note: RenderedNote) -> list[ValidationIssue]:
    raise NotImplementedError  # M3


def export_review_sample(notes: list[RenderedNote], fraction: float, seed: int) -> str:
    """Markdown with structure and note side by side, for physician review."""
    raise NotImplementedError  # M3
