"""Bounds engine tests on toy calculators; clinical ones are in test_calculator_bounds.py."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import partial_and_completion

from calc_bounds.bounds import (
    Exact,
    Interval,
    decision_relevant_missing,
    from_extractions,
    is_determined,
    score_bounds,
    voi_order,
)
from calc_bounds.calculators import Calculator, DecisionCategory
from calc_bounds.calculators.base import ge, gt
from calc_bounds.types import (
    Absent,
    BoolDomain,
    EvidenceSpan,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    Present,
    Unknown,
)

A, B, C = (ParameterSpec(id=n, label=n, domain=BoolDomain()) for n in "abc")
LEVEL = ParameterSpec(id="level", label="level", domain=OrdinalDomain(levels=("0", "1", "2")))
X = ParameterSpec(
    id="x", label="x", domain=NumericDomain(unit="u", lo=0, hi=40), absent_means=(0, 9)
)
Y = ParameterSpec(id="y", label="y", domain=NumericDomain(unit="u", lo=1, hi=10), negatable=False)
Z = ParameterSpec(id="z", label="z", domain=NumericDomain(unit="u", lo=1, hi=10), negatable=False)

TWO_CATS = (
    DecisionCategory(name="low", lo=-math.inf, hi=2),
    DecisionCategory(name="high", lo=2, hi=math.inf),
)

# Three booleans, high if >= 2: no single param narrows the category set on its own.
MAJORITY = Calculator(
    id="majority",
    name="majority",
    citation="toy",
    parameters=(A, B, C),
    categories=TWO_CATS,
    score=lambda v: int(v["a"]) + int(v["b"]) + int(v["c"]),
)

# Ordinal + step numeric: x >= 10 adds 1, x > 20 adds another 1.
STEP = Calculator(
    id="step",
    name="step",
    citation="toy",
    parameters=(A, LEVEL, X),
    categories=(
        DecisionCategory(name="low", lo=-math.inf, hi=2),
        DecisionCategory(name="mid", lo=2, hi=4),
        DecisionCategory(name="high", lo=4, hi=math.inf),
    ),
    score=lambda v: int(v["a"]) + v["level"] + (v["x"] >= 10) + (v["x"] > 20),
    cuts={"x": (ge(10), gt(20))},
)

# Continuous and monotone (increasing in y, decreasing in z), with a discrete factor.
CONT = Calculator(
    id="cont",
    name="cont",
    citation="toy",
    parameters=(A, Y, Z),
    categories=(
        DecisionCategory(name="low", lo=-math.inf, hi=1),
        DecisionCategory(name="mid", lo=1, hi=3),
        DecisionCategory(name="high", lo=3, hi=math.inf),
    ),
    score=lambda v: v["y"] / v["z"] * (0.5 if v["a"] else 1.0),
    continuous=frozenset({"y", "z"}),
)

TOYS = [MAJORITY, STEP, CONT]


def test_step_cut_semantics() -> None:
    # x == 10 counts (>=), x == 20 does not (>).
    assert score_bounds(
        STEP, {"a": Exact(value=False), "level": Exact(value=0), "x": Exact(value=10)}
    ).scores == {1}
    assert score_bounds(
        STEP, {"a": Exact(value=False), "level": Exact(value=0), "x": Exact(value=20)}
    ).scores == {1}
    b = score_bounds(
        STEP, {"a": Exact(value=False), "level": Exact(value=0), "x": Interval(lo=9, hi=20)}
    )
    assert b.scores == {0, 1}
    b = score_bounds(
        STEP, {"a": Exact(value=False), "level": Exact(value=0), "x": Interval(lo=20, hi=20.5)}
    )
    assert b.scores == {1, 2}


def test_majority_unknowns_all_relevant_even_though_none_narrows_alone() -> None:
    assert not is_determined(MAJORITY, {})
    assert decision_relevant_missing(MAJORITY, {}) == ["a", "b", "c"]
    known = {"a": Exact(value=True), "b": Exact(value=True)}
    assert is_determined(MAJORITY, known)
    assert decision_relevant_missing(MAJORITY, known) == []


def test_irrelevant_param_not_asked() -> None:
    # level = 2 and a = True -> score >= 3; x decides mid vs high only when... x matters.
    known = {"a": Exact(value=True), "level": Exact(value=2)}
    assert decision_relevant_missing(STEP, known) == ["x"]
    # level = 0, a = False: score in {0,1,2} -> x matters (x > 20 -> 2 = mid).
    known = {"a": Exact(value=False), "level": Exact(value=0), "x": Interval(lo=0, hi=9)}
    assert is_determined(STEP, known)


def test_continuous_bounds() -> None:
    b = score_bounds(CONT, {"a": Exact(value=False)})
    assert (b.lo, b.hi, b.scores) == (0.1, 10.0, None)
    assert b.categories == {"low", "mid", "high"}
    b = score_bounds(
        CONT, {"a": Exact(value=False), "z": Exact(value=2), "y": Interval(lo=2, hi=5)}
    )
    assert b.categories == {"mid"}


def test_voi_prefers_param_that_resolves() -> None:
    # a=True, level=1: score = 2 + x-points in {2,3,4}. Only x is relevant.
    known = {"a": Exact(value=True), "level": Exact(value=1)}
    assert voi_order(STEP, known, {}) == ["x"]
    # MAJORITY with a known True: b or c True resolves; belief says c is likely True.
    known = {"a": Exact(value=True)}
    order = voi_order(MAJORITY, known, {"b": {True: 0.1, False: 0.9}, "c": {True: 0.9, False: 0.1}})
    assert order == ["c", "b"]


def _span() -> EvidenceSpan:
    return EvidenceSpan(start=0, end=1, text="x")


def test_from_extractions_tristate() -> None:
    ex = {
        "a": Absent(confidence=0.9, confidence_source="oracle", evidence=_span()),
        "level": Present(value=2, confidence=1, confidence_source="oracle", evidence=_span()),
        "x": Unknown(),
    }
    known = from_extractions(STEP, ex)
    assert known == {"a": Exact(value=False), "level": Exact(value=2)}
    # Unknown never becomes Absent unless in the binary ablation.
    assert "x" not in known
    assert from_extractions(STEP, ex, binary=True)["x"] == Interval(lo=0, hi=9)


def test_from_extractions_non_negatable() -> None:
    with pytest.raises(ValueError):
        from_extractions(
            CONT, {"y": Absent(confidence=1, confidence_source="oracle", evidence=_span())}
        )
    # Non-negatable params stay unknown even in the binary ablation.
    assert "y" not in from_extractions(CONT, {}, binary=True)


def test_from_extractions_normalizes_units() -> None:
    from calc_bounds.types import NumericDomain as ND

    urea = ParameterSpec(
        id="urea", label="urea", domain=ND(unit="mmol/L", lo=0, hi=60), absent_means=(2, 7)
    )
    calc = Calculator(
        id="u",
        name="u",
        citation="toy",
        parameters=(urea,),
        categories=TWO_CATS,
        score=lambda v: float(v["urea"] > 7),
        cuts={"urea": (gt(7),)},
    )
    ex = {
        "urea": Present(
            value=28.014,
            unit="bun_mg/dL",
            confidence=1,
            confidence_source="oracle",
            evidence=_span(),
        )
    }
    c = from_extractions(calc, ex)["urea"]
    assert isinstance(c, Exact)
    assert c.value == pytest.approx(10.0)


# --- Property tests (mandatory) -----------------------------------------------------------------


@pytest.mark.parametrize("calc", TOYS, ids=lambda c: c.id)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_bounds_sound(calc: Calculator, data: st.DataObject) -> None:
    known, completion = data.draw(partial_and_completion(calc))
    score, category = calc.evaluate(completion)
    b = score_bounds(calc, known)
    assert b.lo - 1e-9 <= score <= b.hi + 1e-9
    if b.scores is not None:
        assert score in b.scores
    assert category in b.categories
    if is_determined(calc, known):
        assert b.categories == {category}


@pytest.mark.parametrize("calc", TOYS, ids=lambda c: c.id)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_irrelevant_params_cannot_change_category(calc: Calculator, data: st.DataObject) -> None:
    from strategies import value_of

    known, completion = data.draw(partial_and_completion(calc))
    relevant = set(decision_relevant_missing(calc, known))
    for p in calc.parameters:
        if p.id in known or p.id in relevant:
            continue
        other = data.draw(value_of(calc, p))
        _, before = calc.evaluate(completion)
        _, after = calc.evaluate({**completion, p.id: other})
        assert before == after, f"{p.id} changed category but was deemed irrelevant"
