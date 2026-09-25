"""Deterministic simulated clinician. Answers from hidden truth; no LLM involved."""

from typing import Literal

from pydantic import BaseModel

from calc_bounds.types import ParamId, Value


class Answer(BaseModel):
    param: ParamId
    status: Literal["answered", "not_available"]
    value: Value | None = None


class SimulatedClinician:
    """Answers questions from `truth`; params in `unavailable` return not_available
    (e.g. troponin not yet resulted). Logs every question in `log`."""

    def __init__(self, truth: dict[ParamId, Value], unavailable: frozenset[ParamId]) -> None:
        self.truth = truth
        self.unavailable = unavailable
        self.log: list[Answer] = []

    def ask(self, param: ParamId) -> Answer:
        raise NotImplementedError  # M2
