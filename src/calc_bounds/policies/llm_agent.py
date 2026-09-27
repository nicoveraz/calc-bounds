"""S2: end-to-end LLM agent with `calculate` and `ask_clinician` tools.

Implemented as a provider-agnostic JSON-action loop (works with `claude -p`, which has no
custom tools): each turn the model returns one action -
  {"action": "ask", "parameter": <id>, "question": <text>}   -> the simulated clinician answers
  {"action": "calculate", "values": {<id>: <value>, ...}}     -> our calculator code answers
  {"action": "answer", "category": <category | "cannot_determine">}
The clinician is deterministic and answers by parameter id (the free-text question is only
logged), as agreed in M0. Every turn is an LLM call cached by its full transcript, so runs are
resumable and replayable. Premature commitment is judged against the tri-state knowledge
actually available (documented facts + answers), exactly as for the code policies.
"""

import json
from typing import Any

from calc_bounds.bounds import (
    Exact,
    decision_relevant_missing,
    from_extractions,
    score_bounds,
)
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import ExtractionResult, Extractor
from calc_bounds.extraction.oracle import oracle_extractions
from calc_bounds.llm import LLM, LLMRequest, Usage
from calc_bounds.policies.base import Step, Trace
from calc_bounds.render.facts import fmt
from calc_bounds.simulator import SimulatedClinician
from calc_bounds.types import BoolDomain, NumericDomain, OrdinalDomain, ParamId, Value

AGENT_PROMPT_VERSION = "a1"
ABSTAIN = "cannot_determine"

AGENT_SYSTEM = """\
You are a clinical decision-support agent working with an emergency clinician. You determine \
a patient's decision category for one clinical calculator. You can ask the treating clinician \
for any parameter, and you can run the calculator. Ask only what you need; answer when you \
are confident. Each turn, output exactly one JSON action and nothing else."""


def _param_line(p: Any) -> str:
    match p.domain:
        case BoolDomain():
            kind = "yes/no"
        case OrdinalDomain(levels=levels):
            kind = "one of: " + ", ".join(levels)
        case NumericDomain(unit=unit):
            kind = f"number in {unit}"
    return f"- {p.id}: {p.label} [{kind}]"


def initial_prompt(note: str, calc: Calculator) -> str:
    params = "\n".join(_param_line(p) for p in calc.parameters)
    cats = ", ".join(calc.category_names())
    return f"""\
Calculator: {calc.name}
Decision categories: {cats}

Clinical note:
<note>
{note}
</note>

Calculator parameters (ids you can ask about or pass to calculate):
{params}

Actions (reply with one JSON object per turn):
1. {{"action": "ask", "parameter": "<id>", "question": "<your question to the clinician>"}}
   The clinician answers from the patient (or says the value is not available).
2. {{"action": "calculate", "values": {{"<id>": <value>, ...}}}}
   Runs the calculator. Give every parameter: yes/no as true/false, graded ones as the level \
name, numbers in the unit shown. Returns the score and category, or the missing parameters.
3. {{"action": "answer", "category": "<one of: {cats}, {ABSTAIN}>"}}
   Your final decision category. Use "{ABSTAIN}" only if it truly cannot be determined."""


def action_schema(calc: Calculator) -> dict[str, Any]:
    ids = [p.id for p in calc.parameters]
    value: dict[str, Any] = {"anyOf": [{"type": "number"}, {"type": "boolean"}, {"type": "string"}]}
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["ask", "calculate", "answer"]},
            "parameter": {"anyOf": [{"type": "string", "enum": ids}, {"type": "null"}]},
            "question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "values": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": dict.fromkeys(ids, value),
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "category": {
                "anyOf": [
                    {"type": "string", "enum": [*calc.category_names(), ABSTAIN]},
                    {"type": "null"},
                ]
            },
        },
        "required": ["action", "parameter", "question", "values", "category"],
        "additionalProperties": False,
    }


def _coerce(calc: Calculator, values: dict[str, Any]) -> tuple[dict[ParamId, Value], list[str]]:
    out: dict[ParamId, Value] = {}
    problems: list[str] = []
    for p in calc.parameters:
        if p.id not in values or values[p.id] is None:
            problems.append(f"missing {p.id}")
            continue
        v = values[p.id]
        match p.domain:
            case BoolDomain():
                if isinstance(v, str):
                    v = v.strip().lower() in ("true", "yes", "1")
                out[p.id] = bool(v)
            case OrdinalDomain(levels=levels):
                if v not in levels:
                    problems.append(f"{p.id} must be one of {list(levels)}")
                else:
                    out[p.id] = levels.index(v)
            case NumericDomain():
                try:
                    out[p.id] = float(v)
                except (TypeError, ValueError):
                    problems.append(f"{p.id} must be a number")
    return out, problems


def _describe_answer(calc: Calculator, pid: ParamId, value: Value | None) -> str:
    if value is None:
        return f"Clinician: {pid} is not available."
    p = calc.param(pid)
    match p.domain:
        case BoolDomain():
            text = "yes" if value else "no"
        case OrdinalDomain(levels=levels):
            text = levels[int(value)]
        case NumericDomain(unit=unit):
            text = f"{fmt(value)} {unit}"
    return f"Clinician: {pid} = {text}."


def reference_known(case: PatientCase, calc: Calculator) -> dict:
    """What the note actually determines: documented facts, except comorbidities conveyed only
    through a medication (e.g. lisinopril), which the note does not determine."""
    from calc_bounds.cohort import TrapKind

    known = from_extractions(calc, oracle_extractions(calc.parameters, case.truth, case.documented))
    for t in case.traps:
        if t.kind == TrapKind.COMORBIDITY_VIA_MEDICATION:
            known.pop(t.param, None)
    return known


class LLMAgentPolicy:
    id = "s2_llm_agent"

    def __init__(
        self,
        llm: LLM,
        provider: str,
        request_params: dict[str, Any],
        model: str,
        kind: str,
        max_turns: int = 12,
        max_invalid: int = 3,
    ) -> None:
        self.llm = llm
        self.provider = provider
        self.params = request_params
        self.model = model
        self.kind = kind
        self.max_turns = max_turns
        self.max_invalid = max_invalid

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace:
        # What is truly known: documented facts (+ answers), for premature-commitment and
        # irrelevant-question accounting. The agent itself never sees this structure.
        known = reference_known(case, calc)
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt(note, calc)}]
        steps: list[Step] = []
        usage = Usage()
        category: str | None = None
        invalid = 0
        for _ in range(self.max_turns):
            req = LLMRequest(
                provider=self.kind,
                model=self.model,
                system=AGENT_SYSTEM,
                messages=messages,
                params=self.params | {"format": action_schema(calc)},
            )
            resp = self.llm.complete(req, provider=self.provider, stage="agent:s2")
            usage.input_tokens += resp.usage.input_tokens
            usage.output_tokens += resp.usage.output_tokens
            usage.latency_s += resp.usage.latency_s
            messages.append({"role": "assistant", "content": resp.text})
            try:
                action = json.loads(resp.text[resp.text.find("{") : resp.text.rfind("}") + 1])
                kind = action["action"]
            except (json.JSONDecodeError, KeyError, TypeError):
                invalid += 1
                if invalid >= self.max_invalid:
                    break
                messages.append(
                    {"role": "user", "content": "Invalid action; reply with one JSON action."}
                )
                continue
            if kind == "answer":
                cat = action.get("category")
                category = cat if cat in calc.category_names() else None
                break
            if kind == "ask" and action.get("parameter") in {p.id for p in calc.parameters}:
                pid = action["parameter"]
                relevant = decision_relevant_missing(calc, known)
                bounds = score_bounds(calc, known)
                answer = clinician.ask(pid)
                steps.append(
                    Step(
                        bounds=bounds,
                        relevant=relevant,
                        question=pid,
                        reason="llm_choice",
                        answer=answer,
                    )
                )
                if answer.status == "answered":
                    known[pid] = Exact(value=answer.value)  # type: ignore[arg-type]
                messages.append(
                    {
                        "role": "user",
                        "content": _describe_answer(
                            calc, pid, answer.value if answer.status == "answered" else None
                        ),
                    }
                )
                continue
            if kind == "calculate":
                values, problems = _coerce(calc, action.get("values") or {})
                if problems:
                    reply = "Calculator error: " + "; ".join(problems)
                else:
                    score, cat = calc.evaluate(values)
                    reply = f"Calculator: score {fmt(score)}, category {cat}."
                messages.append({"role": "user", "content": reply})
                continue
            invalid += 1
            if invalid >= self.max_invalid:
                break
            messages.append(
                {"role": "user", "content": "Invalid action; reply with one JSON action."}
            )
        final = score_bounds(calc, known)
        return Trace(
            case_id=case.case_id,
            policy=self.id,
            calculator=calc.id,
            extraction=ExtractionResult(case_id=case.case_id, values={}),
            initial_known={},
            steps=steps,
            final_bounds=final,
            final_score=None,
            final_category=category,
            committed_while_undetermined=category is not None and len(final.categories) > 1,
            usage=usage,
        )
