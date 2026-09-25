"""Confidence calibration (temperature scaling, isotonic). Fitted on the dev split only."""

from typing import Literal, Protocol


class Calibrator(Protocol):
    method: Literal["temperature", "isotonic"]

    def fit(self, confidences: list[float], correct: list[bool]) -> None: ...

    def transform(self, confidences: list[float]) -> list[float]: ...
