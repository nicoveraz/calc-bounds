import pytest

from calc_bounds.calculators import REGISTRY
from calc_bounds.cohort import PatientCase, generate_cohort
from calc_bounds.config import CohortConfig
from calc_bounds.extraction.oracle import OracleExtractor
from calc_bounds.policies import POLICIES, Trace
from calc_bounds.simulator import SimulatedClinician, draw_unavailable
from calc_bounds.types import DocumentedState

CFG = CohortConfig(
    n_per_calculator=30, missingness=0.4, normal_as_negation_rate=0.5, undetermined_fraction=0.6
)


@pytest.fixture(scope="module")
def cohort() -> list[PatientCase]:
    return generate_cohort(list(REGISTRY.values()), CFG, seed=11)


def _run(policy: str, case: PatientCase, unavailable: frozenset[str] = frozenset()) -> Trace:
    calc = REGISTRY[case.calculator]
    extractor = OracleExtractor({case.case_id: case})
    return POLICIES[policy].run(
        case, "", calc, extractor, SimulatedClinician(case.truth, unavailable)
    )


def test_simulator() -> None:
    c = SimulatedClinician({"a": True, "b": 3}, frozenset({"b"}))
    assert c.ask("a").value is True
    assert c.ask("b").status == "not_available"
    assert [a.param for a in c.log] == ["a", "b"]
    with pytest.raises(KeyError):
        c.ask("zzz")


def test_draw_unavailable_deterministic() -> None:
    params = [f"p{i}" for i in range(200)]
    a = draw_unavailable(params, {p: 0.3 for p in params}, seed=5)
    assert a == draw_unavailable(list(reversed(params)), {p: 0.3 for p in params}, seed=5)
    assert 30 < len(a) < 90
    assert draw_unavailable(params, {}, seed=5) == frozenset()


def test_oracle_s1_s3_always_correct_without_unavailable(cohort: list[PatientCase]) -> None:
    for case in cohort:
        s1, s3 = _run("s1_ask_all", case), _run("s3_bounds", case)
        assert s1.final_category == s3.final_category == case.true_category
        assert s3.n_questions <= s1.n_questions


def test_s1_asks_every_undocumented_param(cohort: list[PatientCase]) -> None:
    for case in cohort:
        t = _run("s1_ask_all", case)
        undocumented = [
            p for p, s in case.documented.items() if s == DocumentedState.NOT_DOCUMENTED
        ]
        assert sorted(s.question for s in t.steps) == sorted(undocumented)
        # "Stated normal" numeric values are intervals, so the exact score may stay unknown.
        assert t.final_score in (None, case.true_score)
        assert t.final_category == case.true_category


def test_s3_asks_only_relevant_and_nothing_when_determined(cohort: list[PatientCase]) -> None:
    for case in cohort:
        t = _run("s3_bounds", case)
        for s in t.steps:
            assert s.relevant is not None and s.question in s.relevant
            assert len(s.bounds.categories) > 1
        if case.determined_from_note:
            assert t.n_questions == 0


def test_unknown_never_becomes_absent_except_s3_bin(cohort: list[PatientCase]) -> None:
    for case in cohort:
        undocumented = {
            p for p, s in case.documented.items() if s == DocumentedState.NOT_DOCUMENTED
        }
        assert not undocumented & set(_run("s3_bounds", case).initial_known)
        assert not undocumented & set(_run("s1_ask_all", case).initial_known)
    # The binary ablation does fill undocumented (negatable) params with "absent".
    filled = sum(
        len(
            {p for p, s in c.documented.items() if s == DocumentedState.NOT_DOCUMENTED}
            & set(_run("s3_bin", c).initial_known)
        )
        for c in cohort
    )
    assert filled > 0


def test_unavailable_answer_leads_to_abstention_not_a_guess(cohort: list[PatientCase]) -> None:
    abstained = 0
    for case in cohort:
        calc = REGISTRY[case.calculator]
        t = _run("s3_bounds", case, frozenset(p.id for p in calc.parameters))
        if t.final_category is None:
            abstained += 1
            assert len(t.final_bounds.categories) > 1
            assert not t.committed_while_undetermined
        else:
            assert t.final_category == case.true_category
    assert abstained == sum(not c.determined_from_note for c in cohort)
