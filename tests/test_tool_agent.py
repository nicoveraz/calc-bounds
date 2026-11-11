"""S5 (bounded-calcs tool agent) with scripted models: no LLM calls."""

import json
import re
from importlib import resources
from typing import Any

import pytest

from calc_bounds.bounds import Exact, Interval
from calc_bounds.calculators import REGISTRY
from calc_bounds.cohort import PatientCase, generate_cohort
from calc_bounds.config import CohortConfig
from calc_bounds.llm import LLMRequest, LLMResponse, Usage
from calc_bounds.policies import Trace
from calc_bounds.policies.llm_agent import ABSTAIN, reference_known
from calc_bounds.policies.tool_agent import TOOL_RULES, ToolAgentPolicy, audit_finding
from calc_bounds.simulator import ClinicianNoise, SimulatedClinician
from calc_bounds.types import BoolDomain, NumericDomain, OrdinalDomain

CFG = CohortConfig(
    n_per_calculator=30, missingness=0.4, normal_as_negation_rate=0.5, undetermined_fraction=0.6
)
TOOL_CALCS = ["heart", "curb65", "qsofa", "perc", "wells_pe"]
SITE = {"urea": (2.5, 7.0)}  # the cohort's "stated normal" urea range


@pytest.fixture(scope="module")
def cohort() -> list[PatientCase]:
    return generate_cohort([REGISTRY[c] for c in TOOL_CALCS], CFG, seed=11)


def documented_findings(case: PatientCase) -> dict[str, Any]:
    """What an honest reader sends: documented values, "normal" for stated-normal numbers."""
    calc = REGISTRY[case.calculator]
    out: dict[str, Any] = {}
    for pid, c in reference_known(case, calc).items():
        domain = calc.param(pid).domain
        if isinstance(c, Interval):
            out[pid] = "normal"
        elif isinstance(domain, BoolDomain):
            out[pid] = "present" if c.value else "absent"
        elif isinstance(domain, OrdinalDomain):
            out[pid] = domain.levels[int(c.value)]
        else:
            out[pid] = float(c.value)
    return out


class ScriptedModel:
    """Plays the agent. `honest` follows the tool's rules; otherwise it fills every
    undocumented yes/no item as absent before the first call."""

    def __init__(self, case: PatientCase, honest: bool = True) -> None:
        self.findings = documented_findings(case)
        if not honest:
            for p in REGISTRY[case.calculator].parameters:
                if isinstance(p.domain, BoolDomain):
                    self.findings.setdefault(p.id, "absent")
        self.asked: set[str] = set()

    def complete(self, request: LLMRequest, *, provider: str, stage: str = "") -> LLMResponse:
        last = request.messages[-1]["content"]
        action: dict[str, Any]
        if last.startswith("Tool result: "):
            result = json.loads(last.removeprefix("Tool result: "))
            pending = [d["item"] for d in result["decisive_unknowns"]]
            pending += result["jointly_relevant_unknowns"]
            pending = [p for p in pending if p not in self.asked]
            if result["settled"]:
                action = {"action": "answer", "category": result["category"]}
            elif pending:
                self.asked.add(pending[0])
                action = {"action": "ask", "item": pending[0], "question": "?"}
            else:
                action = {"action": "answer", "category": ABSTAIN}
        else:
            if m := re.match(r"Clinician: (\w+) is between (\S+) and (\S+)", last):
                self.findings[m[1]] = {"min": float(m[2]), "max": float(m[3])}
            elif m := re.match(r"Clinician: (\w+) = (.+)\.$", last):
                value: Any = m[2]
                if re.fullmatch(r"-?\d+(\.\d+)?( \S+)?", value):  # a number with its unit
                    value = float(value.split()[0])
                self.findings[m[1]] = value
            action = {"action": "evaluate", "findings": self.findings}
        return LLMResponse(text=json.dumps(action), usage=Usage(), stop_reason="end_turn", raw={})


def _run(case: PatientCase, *, honest: bool = True, noise: ClinicianNoise | None = None) -> Trace:
    calc = REGISTRY[case.calculator]
    policy = ToolAgentPolicy(
        ScriptedModel(case, honest),  # type: ignore[arg-type]
        "scripted",
        {},
        "scripted",
        "session",
        normal_ranges=SITE,
        site_units={"urea": "mmol/L"},  # the scripted model works in the cohort's own units
        max_turns=60,  # the scripted model re-evaluates after every answer
    )
    clinician = SimulatedClinician(
        case.truth, specs={p.id: p for p in calc.parameters}, noise=noise, seed=3
    )
    return policy.run(case, "note", calc, None, clinician)  # type: ignore[arg-type]


def test_honest_agent_is_right_and_clean(cohort: list[PatientCase]) -> None:
    for case in cohort:
        t = _run(case)
        a = t.tool_audit
        assert a is not None and a.calls >= 1 and a.tool_errors == 0
        assert t.final_category == case.true_category, case.case_id
        assert not a.unsupported and not a.wrong, (case.case_id, a)
        assert not a.answered_against_tool and not t.committed_while_undetermined
        if case.determined_from_note:
            assert t.n_questions == 0


def test_filling_undocumented_items_is_flagged(cohort: list[PatientCase]) -> None:
    flagged = 0
    for case in cohort:
        t = _run(case, honest=False)
        calc = REGISTRY[case.calculator]
        known = reference_known(case, calc)
        filled = {
            p.id for p in calc.parameters if isinstance(p.domain, BoolDomain) and p.id not in known
        }
        assert t.tool_audit is not None
        assert filled <= set(t.tool_audit.unsupported) | set(t.tool_audit.wrong)
        flagged += bool(t.tool_audit.unsupported)
    assert flagged > len(cohort) // 3


def test_vague_answers_reach_the_tool_as_ranges(cohort: list[PatientCase]) -> None:
    noise = ClinicianNoise(vague_rate=1.0, vague_width=0.1)
    for case in cohort:
        t = _run(case, noise=noise)
        assert t.tool_audit is not None
        assert not t.tool_audit.wrong and not t.tool_audit.unsupported, case.case_id
        assert not t.committed_while_undetermined
        assert t.final_category in (case.true_category, None)


def test_rules_text_matches_the_installed_tool() -> None:
    source = (resources.files("bounded_calcs") / "mcp_server.py").read_text()
    assert TOOL_RULES in source


def test_audit_finding() -> None:
    curb = REGISTRY["curb65"]
    assert isinstance(curb.param("urea").domain, NumericDomain)
    normal = (2.5, 7.0)
    ok = [
        ("confusion", "absent", Exact(value=False)),
        ("urea", 9.0, Exact(value=9.0)),
        ("urea", 8.92, Exact(value=9.0)),  # converted from another unit
        ("urea", {"min": 8}, Exact(value=9.0)),
        ("urea", "normal", Interval(lo=2.5, hi=7.0)),
        ("urea", {"min": 5, "max": 9}, Interval(lo=2.5, hi=7.0)),
    ]
    for pid, value, ref in ok:
        assert audit_finding(curb, pid, value, ref, normal, False) == "ok", (pid, value)
    assert audit_finding(curb, "confusion", "absent", None, None, False) == "unsupported"
    assert audit_finding(curb, "confusion", "present", Exact(value=False), None, False) == "wrong"
    assert audit_finding(curb, "urea", 12.0, Exact(value=9.0), normal, False) == "wrong"
    assert audit_finding(curb, "urea", "normal", Exact(value=9.0), normal, False) == "wrong"
    assert audit_finding(curb, "urea", 5.0, Interval(lo=2.5, hi=7.0), normal, False) == "wrong"
    # A comorbidity implied only by a medication: present is acceptable, absent is not.
    assert audit_finding(curb, "confusion", "present", None, None, True) == "ok"
    assert audit_finding(curb, "confusion", "absent", None, None, True) == "wrong"


def test_judgment_confirmations_are_counted_apart() -> None:
    from calc_bounds.policies.base import Step
    from calc_bounds.simulator import Answer
    from calc_bounds.tooleval import judgment_confirmations

    def step(pid: str, relevant: list[str]) -> Step:
        answer = Answer(param=pid, status="answered", value=1)
        return Step(
            bounds=None, relevant=relevant, question=pid, reason="llm_choice", answer=answer
        )

    trace = Trace(
        case_id="heart-x",
        policy="s5_tool_agent",
        calculator="heart",
        extraction={"case_id": "heart-x", "values": {}},  # type: ignore[arg-type]
        initial_known={},
        steps=[
            step("heart_history", ["heart_troponin"]),  # judgment, already known: confirmation
            step("heart_history", ["heart_history"]),  # judgment, relevant: an ordinary question
            step("age", ["heart_troponin"]),  # not a judgment: plainly irrelevant
        ],
        final_bounds=None,
        final_score=None,
        final_category=None,
        committed_while_undetermined=False,
    )
    assert judgment_confirmations(trace) == 1
