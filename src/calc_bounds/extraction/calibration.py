"""Confidence calibration (temperature scaling, isotonic). Fitted on the dev split only.

Calibration targets are Present/Absent claims: a claim is correct when it matches the
documented state (and, for Present, the true value/level). Unknown carries no confidence.
"""

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator(BaseModel):
    method: Literal["none", "temperature", "isotonic"]
    temperature: float = 1.0
    iso_x: list[float] = []
    iso_y: list[float] = []

    def transform(self, confidences: list[float] | np.ndarray) -> np.ndarray:
        c = np.asarray(confidences, dtype=float)
        match self.method:
            case "none":
                return c
            case "temperature":
                return 1 / (1 + np.exp(-_logit(c) / self.temperature))
            case "isotonic":
                return np.interp(c, self.iso_x, self.iso_y)


def fit(
    method: Literal["none", "temperature", "isotonic"],
    confidences: list[float],
    correct: list[bool],
) -> Calibrator:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if method == "none" or len(c) == 0:
        return Calibrator(method="none")
    if method == "temperature":
        z = _logit(c)

        def nll(t: float) -> float:
            p = np.clip(1 / (1 + np.exp(-z / t)), _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        t = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x
        return Calibrator(method="temperature", temperature=float(t))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(c, y)
    return Calibrator(
        method="isotonic",
        iso_x=[float(x) for x in iso.X_thresholds_],
        iso_y=[float(v) for v in iso.y_thresholds_],
    )


def brier(confidences: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((np.asarray(confidences) - np.asarray(correct, dtype=float)) ** 2))


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return math.nan
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def reliability(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean confidence, accuracy, count) per non-empty bin."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    return [
        (float(c[bins == b].mean()), float(y[bins == b].mean()), int((bins == b).sum()))
        for b in range(n_bins)
        if (bins == b).any()
    ]
