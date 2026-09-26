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


def test_s3_bin_premature_commitment_is_measured(cohort: list[PatientCase]) -> None:
    traces = [_run("s3_bin", c) for c in cohort]
    premature = [t for t in traces if t.committed_while_undetermined]
    assert premature, "binary schema should sometimes commit while truly undetermined"
    for t in [_run("s3_bounds", c) for c in cohort]:
        assert not t.committed_while_undetermined


def test_prior_beliefs_are_distributions() -> None:
    from calc_bounds.cohort.priors import DEFAULT_PRIORS
    from calc_bounds.policies.voi_echo import prior_beliefs

    for calc_id, calc in REGISTRY.items():
        beliefs = prior_beliefs(calc, DEFAULT_PRIORS[calc_id])
        for pid, b in beliefs.items():
            assert abs(sum(b.values()) - 1) < 1e-9, (calc_id, pid)
            assert all(w >= 0 for w in b.values())


def _s4(threshold: float = 0.9):
    from calc_bounds.cohort.priors import DEFAULT_PRIORS
    from calc_bounds.extraction.calibration import Calibrator
    from calc_bounds.policies.voi_echo import VoiEchoPolicy

    return VoiEchoPolicy(DEFAULT_PRIORS, Calibrator(method="none"), threshold)


def test_s4_with_oracle_matches_s3_accuracy_and_never_echoes(cohort: list[PatientCase]) -> None:
    s4 = _s4()
    for case in cohort:
        calc = REGISTRY[case.calculator]
        t = s4.run(
            case, "", calc, OracleExtractor({case.case_id: case}), SimulatedClinician(case.truth)
        )
        assert t.final_category == case.true_category
        assert all(s.reason == "decision_relevant" for s in t.steps)  # oracle confidence = 1
        assert t.n_questions <= len(decision_relevant_missing_at_start(case))


def decision_relevant_missing_at_start(case: PatientCase) -> list[str]:
    from calc_bounds.bounds import decision_relevant_missing, from_extractions
    from calc_bounds.extraction.oracle import oracle_extractions

    calc = REGISTRY[case.calculator]
    ex = oracle_extractions(calc.parameters, case.truth, case.documented)
    unknown = [p.id for p in calc.parameters if ex[p.id].kind == "unknown"]
    return unknown if decision_relevant_missing(calc, from_extractions(calc, ex)) else []


def test_s4_echo_catches_low_confidence_wrong_value() -> None:
    from calc_bounds.extraction import ExtractionResult
    from calc_bounds.extraction.oracle import PrecomputedExtractor
    from calc_bounds.types import EvidenceSpan, Present

    calc = REGISTRY["qsofa"]
    truth = {"resp_rate": 24.0, "altered_mentation": True, "sbp": 120.0}  # score 2: positive
    case = PatientCase(
        case_id="q",
        seed=0,
        calculator="qsofa",
        truth=truth,
        documented={
            "resp_rate": DocumentedState.POSITIVE,
            "altered_mentation": DocumentedState.POSITIVE,
            "sbp": DocumentedState.POSITIVE,
        },
        true_score=2,
        true_category="positive",
        determined_from_note=True,
    )
    ev = EvidenceSpan(start=0, end=1, text="x")
    wrong = ExtractionResult(
        case_id="q",
        values={
            "resp_rate": Present(
                value=18.0,
                unit="/min",
                confidence=0.4,
                confidence_source="self_reported",
                evidence=ev,
            ),  # misread
            "altered_mentation": Present(
                value=True, confidence=0.99, confidence_source="self_reported", evidence=ev
            ),
            "sbp": Present(
                value=120.0,
                unit="mmHg",
                confidence=0.99,
                confidence_source="self_reported",
                evidence=ev,
            ),
        },
    )
    ext = PrecomputedExtractor("x", {"q": wrong})
    s3 = POLICIES["s3_bounds"].run(case, "", calc, ext, SimulatedClinician(truth))
    assert s3.final_category == "negative" and s3.n_questions == 0  # confidently wrong
    s4 = _s4().run(case, "", calc, ext, SimulatedClinician(truth))
    assert [(s.question, s.reason) for s in s4.steps] == [("resp_rate", "confidence_echo")]
    assert s4.final_category == "positive"


def test_s2_agent_loop_with_scripted_llm(tmp_path) -> None:
    import json

    from calc_bounds.llm import LLM, DiskCache, LLMResponse, Usage
    from calc_bounds.policies.llm_agent import LLMAgentPolicy

    calc = REGISTRY["qsofa"]
    truth = {"resp_rate": 24.0, "altered_mentation": True, "sbp": 120.0}
    case = PatientCase(
        case_id="q",
        seed=0,
        calculator="qsofa",
        truth=truth,
        documented={
            "resp_rate": DocumentedState.POSITIVE,
            "altered_mentation": DocumentedState.NOT_DOCUMENTED,
            "sbp": DocumentedState.POSITIVE,
        },
        true_score=2,
        true_category="positive",
        determined_from_note=False,
    )

    def act(**kw):
        base = {
            "action": None,
            "parameter": None,
            "question": None,
            "values": None,
            "category": None,
        }
        return json.dumps(base | kw)

    script = [
        "not json",
        act(action="ask", parameter="altered_mentation", question="Is the patient confused?"),
        act(action="calculate", values={"resp_rate": 24, "altered_mentation": True}),
        act(action="calculate", values={"resp_rate": 24, "altered_mentation": True, "sbp": 120}),
        act(action="answer", category="positive"),
    ]

    class Scripted:
        def complete(self, request):
            turn = sum(m["role"] == "assistant" for m in request.messages)
            return LLMResponse(text=script[turn], usage=Usage(input_tokens=10, output_tokens=5))

    llm = LLM({"fake": Scripted()}, DiskCache(tmp_path), max_cost_usd=0)
    agent = LLMAgentPolicy(llm, "fake", {}, "m", "fake")
    t = agent.run(case, "RR 24. BP 120/70.", calc, None, SimulatedClinician(truth))
    assert t.final_category == "positive" and not t.committed_while_undetermined
    assert [(s.question, s.reason, s.answer.value) for s in t.steps] == [
        ("altered_mentation", "llm_choice", True)
    ]
    assert t.steps[0].relevant == ["altered_mentation"]
    assert t.usage.input_tokens == 50

    # Answering before asking: committed while undetermined (mentation decides the category).
    script[:] = [act(action="answer", category="negative")]
    t = LLMAgentPolicy(
        LLM({"fake": Scripted()}, DiskCache(tmp_path / "b"), max_cost_usd=0),
        "fake",
        {},
        "m",
        "fake",
    ).run(case, "note", calc, None, SimulatedClinician(truth))
    assert t.final_category == "negative" and t.committed_while_undetermined and t.n_questions == 0
