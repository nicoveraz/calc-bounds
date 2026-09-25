"""Score bounds over partially known parameters, and decision relevance.

Method (exact, no sampling):
  * bool / ordinal params: enumerate their values.
  * step numeric params: enumerate one representative per region between the calculator's
    `cuts`, intersected with the param's allowed interval. The score is constant within a
    region, so this is exact.
  * continuous numeric params: the score is continuous and strictly monotone in each, so for a
    fixed value of everything else, the reachable scores are exactly the closed interval
    between the minimum and maximum over the corners of their box.

`from_extractions` is the only place where tri-state semantics enter bounds; Unknown is never
treated as Absent except with `binary=True` (the S3-bin ablation).
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from calc_bounds.calculators import Calculator
from calc_bounds.calculators.base import Cut
from calc_bounds.types import (
    Absent,
    BoolDomain,
    Extraction,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    ParamId,
    Present,
    Unknown,
    Value,
)
from calc_bounds.units import to_canonical


class Exact(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["exact"] = "exact"
    value: Value


class Interval(BaseModel):
    """Numeric param known to lie in the closed interval [lo, hi] (e.g. 'stated normal')."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["interval"] = "interval"
    lo: float
    hi: float


type Constraint = Annotated[Exact | Interval, Field(discriminator="kind")]
type Partial = Mapping[ParamId, Constraint]
"""A partial assignment. Params absent from the mapping are unknown (full domain)."""


class ScoreBounds(BaseModel):
    lo: float
    hi: float
    scores: frozenset[float] | None
    """Exact achievable score set when enumerable; None when continuous params are free."""
    categories: frozenset[str]


# --- Candidate values per parameter ------------------------------------------------------------


def _interval(p: ParameterSpec, c: Constraint | None) -> tuple[float, float]:
    assert isinstance(p.domain, NumericDomain)
    if c is None:
        return p.domain.lo, p.domain.hi
    if isinstance(c, Exact):
        return float(c.value), float(c.value)
    return c.lo, c.hi


def _region_representatives(cuts: Sequence[Cut], lo: float, hi: float) -> list[float]:
    """One point per nonempty region of the real line split by `cuts`, within [lo, hi]."""
    cuts = sorted(cuts, key=lambda c: c.at)
    # Region i lies between cut i-1 and cut i: (a, a_incl, b, b_incl).
    bounds: list[tuple[float, bool, float, bool]] = []
    prev: tuple[float, bool] = (-math.inf, False)
    for c in cuts:
        bounds.append((prev[0], prev[1], c.at, not c.upper_inclusive))
        prev = (c.at, c.upper_inclusive)
    bounds.append((prev[0], prev[1], math.inf, False))

    reps: list[float] = []
    for a, a_incl, b, b_incl in bounds:
        # Intersect with the closed interval [lo, hi].
        if lo > a or (lo == a and a_incl):
            a, a_incl = lo, True
        if hi < b or (hi == b and b_incl):
            b, b_incl = hi, True
        if a < b:
            reps.append((a + b) / 2 if math.isfinite(a) and math.isfinite(b) else a)
        elif a == b and a_incl and b_incl:
            reps.append(a)
    return reps


def _candidates(calc: Calculator, p: ParameterSpec, c: Constraint | None) -> list[Value]:
    if isinstance(c, Exact):
        return [c.value]
    match p.domain:
        case BoolDomain():
            return [False, True]
        case OrdinalDomain(levels=levels):
            return list(range(len(levels)))
        case NumericDomain():
            lo, hi = _interval(p, c)
            if p.id in calc.continuous:
                return [lo, hi] if lo < hi else [lo]
            return list(_region_representatives(calc.cuts[p.id], lo, hi))
    raise TypeError(p.domain)


# --- Grid evaluation ----------------------------------------------------------------------------


class _Grid(BaseModel):
    """Scores over the candidate grid. Axes follow `calc.parameters` order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[list[Value]]
    scores: np.ndarray
    """Shape = tuple(len(c) for c in candidates)."""


def _grid(calc: Calculator, known: Partial) -> _Grid:
    ids = [p.id for p in calc.parameters]
    unexpected = set(known) - set(ids)
    if unexpected:
        raise ValueError(f"{calc.id}: constraints for unknown params {sorted(unexpected)}")
    cands = [_candidates(calc, p, known.get(p.id)) for p in calc.parameters]
    scores = np.array(
        [calc.score(dict(zip(ids, combo, strict=True))) for combo in itertools.product(*cands)],
        dtype=float,
    ).reshape([len(c) for c in cands])
    return _Grid(candidates=cands, scores=scores)


def _continuous_axes(calc: Calculator) -> tuple[int, ...]:
    return tuple(i for i, p in enumerate(calc.parameters) if p.id in calc.continuous)


def _category_set(calc: Calculator, lo: float, hi: float) -> set[str]:
    """Categories overlapping the closed score interval [lo, hi]."""
    return {c.name for c in calc.categories if c.lo <= hi and lo < c.hi}


def score_bounds(calc: Calculator, known: Partial) -> ScoreBounds:
    g = _grid(calc, known)
    axes = _continuous_axes(calc)
    if not axes or all(len(g.candidates[i]) == 1 for i in axes):
        flat = g.scores.ravel()
        return ScoreBounds(
            lo=float(flat.min()),
            hi=float(flat.max()),
            scores=frozenset(float(s) for s in flat),
            categories=frozenset(calc.category(float(s)) for s in flat),
        )
    # For each combination of the non-continuous params, the reachable scores are the closed
    # interval between min and max over the continuous corners.
    mins = g.scores.min(axis=axes).ravel()
    maxs = g.scores.max(axis=axes).ravel()
    cats: set[str] = set()
    for lo, hi in zip(mins, maxs, strict=True):
        cats |= _category_set(calc, float(lo), float(hi))
    return ScoreBounds(
        lo=float(mins.min()), hi=float(maxs.max()), scores=None, categories=frozenset(cats)
    )


def is_determined(calc: Calculator, known: Partial) -> bool:
    """True iff exactly one decision category is possible."""
    return len(score_bounds(calc, known).categories) == 1


def decision_relevant_missing(calc: Calculator, known: Partial) -> list[ParamId]:
    """Unknown params whose value can change the decision category for some completion of the
    other unknowns (so a param that only matters jointly with others is still relevant).

    Exact for calculators without continuous params. With continuous params (Cockcroft-Gault)
    this is conservative: when undetermined, every unknown param is returned.
    """
    unknown = [i for i, p in enumerate(calc.parameters) if p.id not in known]
    if is_determined(calc, known):
        return []
    if calc.continuous:
        return [calc.parameters[i].id for i in unknown]
    g = _grid(calc, known)
    names = calc.category_names()
    cat_idx = np.vectorize(lambda s: names.index(calc.category(float(s))))(g.scores)
    relevant = []
    for i in unknown:
        if np.any(cat_idx.max(axis=i) != cat_idx.min(axis=i)):
            relevant.append(calc.parameters[i].id)
    return relevant


def resolve_probability(
    calc: Calculator, known: Partial, param: ParamId, belief: Mapping[Value, float] | None
) -> float:
    """P(the category is determined after learning `param` alone), under `belief` (a
    distribution over the param's values; uniform over candidate values if None)."""
    if belief is None:
        cands = _candidates(calc, calc.param(param), None)
        belief = {v: 1.0 / len(cands) for v in cands}
    total = sum(belief.values())
    if total <= 0:
        raise ValueError(f"belief for {param} has no mass")
    return sum(
        w / total
        for v, w in belief.items()
        if w > 0 and is_determined(calc, {**known, param: Exact(value=v)})
    )


def voi_order(
    calc: Calculator, known: Partial, beliefs: Mapping[ParamId, Mapping[Value, float]]
) -> list[ParamId]:
    """Decision-relevant unknowns ranked by P(asking this one alone determines the category)
    (greedy, one step). Ties keep calculator parameter order."""
    relevant = decision_relevant_missing(calc, known)
    probs = {p: resolve_probability(calc, known, p, beliefs.get(p)) for p in relevant}
    return sorted(relevant, key=lambda p: -probs[p])


# --- Tri-state extractions -> partial assignment ------------------------------------------------


def _absent_constraint(p: ParameterSpec) -> Constraint:
    if not p.negatable:
        raise ValueError(f"{p.id} cannot be Absent (not negatable)")
    match p.domain:
        case BoolDomain():
            return Exact(value=False)
        case OrdinalDomain():
            return Exact(value=0)
        case NumericDomain():
            assert p.absent_means is not None
            return Interval(lo=p.absent_means[0], hi=p.absent_means[1])
    raise TypeError(p.domain)


def from_extractions(
    calc: Calculator, extractions: Mapping[ParamId, Extraction], *, binary: bool = False
) -> dict[ParamId, Constraint]:
    """Convert tri-state extractions to a partial assignment (numeric values normalized to
    canonical units by code). Unknown and missing params stay unknown, unless `binary=True`
    (S3-bin ablation): then they are treated as Absent wherever Absent is meaningful."""
    known: dict[ParamId, Constraint] = {}
    for p in calc.parameters:
        e = extractions.get(p.id, Unknown())
        match e:
            case Present(value=v, unit=unit):
                if isinstance(p.domain, NumericDomain):
                    v = to_canonical(p.id, float(v), unit or p.domain.unit)
                known[p.id] = Exact(value=v)
            case Absent():
                known[p.id] = _absent_constraint(p)
            case Unknown():
                if binary and p.negatable:
                    known[p.id] = _absent_constraint(p)
    return known
