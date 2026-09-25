"""Property tests (mandatory): bounds soundness on every clinical calculator."""

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import partial_and_completion, value_of

from calc_bounds.bounds import (
    Exact,
    Interval,
    decision_relevant_missing,
    is_determined,
    score_bounds,
)
from calc_bounds.calculators import REGISTRY, Calculator
from calc_bounds.calculators.base import Cut
from calc_bounds.types import NumericDomain

CALCS = list(REGISTRY.values())


@pytest.mark.parametrize("calc", CALCS, ids=lambda c: c.id)
@settings(max_examples=150, deadline=None)
@given(data=st.data())
def test_every_completion_within_bounds(calc: Calculator, data: st.DataObject) -> None:
    known, completion = data.draw(partial_and_completion(calc))
    score, category = calc.evaluate(completion)
    b = score_bounds(calc, known)
    assert b.lo - 1e-9 <= score <= b.hi + 1e-9
    if b.scores is not None:
        assert score in b.scores
    assert category in b.categories
    if is_determined(calc, known):
        assert b.categories == {category}


@pytest.mark.parametrize("calc", CALCS, ids=lambda c: c.id)
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_irrelevant_params_cannot_change_category(calc: Calculator, data: st.DataObject) -> None:
    known, completion = data.draw(partial_and_completion(calc))
    relevant = set(decision_relevant_missing(calc, known))
    _, before = calc.evaluate(completion)
    for p in calc.parameters:
        if p.id in known or p.id in relevant:
            continue
        _, after = calc.evaluate({**completion, p.id: data.draw(value_of(calc, p))})
        assert before == after, f"{p.id} changed category but was deemed irrelevant"


def _crosses(cut: Cut, lo: float, hi: float) -> bool:
    below = lo < cut.at or (lo == cut.at and not cut.upper_inclusive)
    above = hi > cut.at or (hi == cut.at and cut.upper_inclusive)
    return below and above


@pytest.mark.parametrize("calc", CALCS, ids=lambda c: c.id)
def test_stated_normal_does_not_straddle_thresholds(calc: Calculator) -> None:
    """A param documented as normal must not leave its criterion undetermined."""
    for p in calc.parameters:
        if p.absent_means is None or p.id not in calc.cuts:
            continue
        lo, hi = p.absent_means
        for cut in calc.cuts[p.id]:
            assert not _crosses(cut, lo, hi), f"{calc.id}.{p.id}: normal {lo}-{hi} vs {cut}"


@pytest.mark.parametrize("calc", CALCS, ids=lambda c: c.id)
def test_absent_means_within_domain(calc: Calculator) -> None:
    for p in calc.parameters:
        if p.absent_means is not None:
            assert isinstance(p.domain, NumericDomain)
            assert p.domain.lo <= p.absent_means[0] <= p.absent_means[1] <= p.domain.hi


def test_curb65_example_relevance() -> None:
    calc = REGISTRY["curb65"]
    # Confused 80-year-old with RR 32: already >= 3 -> determined high; nothing to ask.
    known = {"confusion": Exact(value=True), "age": Exact(value=80), "resp_rate": Exact(value=32)}
    assert score_bounds(calc, known).categories == {"high"}
    assert decision_relevant_missing(calc, known) == []
    # 40-year-old, not confused, RR normal, BP normal: score in {0, 1} -> low; urea irrelevant.
    known = {
        "confusion": Exact(value=False),
        "age": Exact(value=40),
        "resp_rate": Interval(lo=12, hi=20),
        "sbp": Interval(lo=101, hi=139),
        "dbp": Interval(lo=61, hi=89),
    }
    assert is_determined(calc, known)
    # Same but 70 years old: score in {1, 2} -> urea is the one question that matters.
    known["age"] = Exact(value=70)
    assert decision_relevant_missing(calc, known) == ["urea"]


def test_cockcroft_gault_determined_with_all_values() -> None:
    calc = REGISTRY["cockcroft_gault"]
    full = {"age": Exact(value=40), "weight": Exact(value=72), "creatinine": Exact(value=1.0)}
    assert score_bounds(calc, full).categories == {"crcl_ge_60"}  # 100 or 85 mL/min
    assert decision_relevant_missing(calc, full) == []


def test_heart_bounds_fast_enough() -> None:
    """Fully unknown HEART is the largest grid (~10k points); keep it interactive."""
    calc = REGISTRY["heart"]
    t = time.perf_counter()
    decision_relevant_missing(calc, {})
    assert time.perf_counter() - t < 2.0
