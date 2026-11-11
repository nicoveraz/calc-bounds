"""S5: an LLM agent that drives the bounded-calcs tool (M7).

Same JSON-action loop as S2, but the agent does not get our calculator: it gets what a client
of the bounded-calcs MCP server gets, the calculator's schema and an `evaluate` call that takes
findings (present / absent / a level / a number / "normal" / a range / unknown) and returns the
tool's JSON result unchanged.
  {"action": "evaluate", "findings": {<item id>: <value>, ...}}  -> bounded_calcs.evaluate
  {"action": "ask", "item": <id>, "question": <text>}            -> the simulated clinician
  {"action": "answer", "category": <category | "cannot_determine">}
The tool is called in-process with the same payloads (no MCP transport, no new dependency).

Every `evaluate` call is audited against what the note documents and what the clinician has
answered: an item asserted without either is a silent fill (`ToolAudit.unsupported`); an item
asserted with a contradicting value is `wrong`. Premature commitment is judged as for S2,
against the knowledge actually available.
"""

import json
import math
from typing import Any, Literal

import bounded_calcs

from calc_bounds.bounds import (
    Constraint,
    Exact,
    Interval,
    decision_relevant_missing,
    score_bounds,
)
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase, TrapKind
from calc_bounds.extraction import ExtractionResult, Extractor
from calc_bounds.llm import LLM, LLMRequest, Usage
from calc_bounds.policies.base import Step, ToolAudit, Trace
from calc_bounds.policies.llm_agent import ABSTAIN, reference_known
from calc_bounds.render.facts import fmt
from calc_bounds.simulator import Answer, SimulatedClinician
from calc_bounds.types import BoolDomain, NumericDomain, OrdinalDomain, ParamId
from calc_bounds.units import UnitError, to_canonical

TOOL_PROMPT_VERSION = "t4"

TOOL_SYSTEM = """\
You are a clinical decision-support agent working with an emergency clinician. You determine \
a patient's decision category for one clinical calculator, using the bounded-calcs tool. You \
can call the tool and you can ask the treating clinician about any item. Each turn, output \
exactly one JSON action and nothing else."""

# Steps 2-4 of the server instructions of the installed bounded-calcs (mcp_server.INSTRUCTIONS),
# verbatim; a test checks this block against the installed package.
TOOL_RULES = """\
2. Read the note and send to `evaluate` only what it actually documents: present, absent, a
   level, or a number. Leave out (or send "unknown") anything the note does not mention.
   Never fill an unmentioned item as absent or normal. If the note says only that a numeric
   item is normal ("vital signs normal"), send "normal" when the schema gives that item a
   normal range, and leave it unknown otherwise. If it gives only bounds ("heart rate over
   100", "SBP in the 90s"), send {"min": x, "max": y}, leaving out the bound it does not
   give. Never invent a number. Send a plain number only in the unit the schema states for
   that item. If the note gives another unit that the schema lists for the item
   (other_units), send {"value": x, "unit": "the note's unit"} and the tool converts it.
   An item the schema marks unit_required always goes that way, with the unit exactly as
   the note gives it (unit_labels says what each listed unit means). Only if the note gives
   the number with no unit at all, send "unit": "not stated" when the schema shows a
   site_unit for the item, and otherwise ask for the unit. Never guess a unit, and convert
   one yourself only when the schema does not list it.
3. If the result is settled, report its category (for a score-only calculator, one whose
   schema has no categories, its score). If not, ask the clinician about the
   decisive unknowns (in the order given), add the answers, and call evaluate again.
4. Items of kind "judgment" (for example HEART history, Wells "alternative diagnosis less
   likely") must come from the clinician, never from your reading of the note."""

# The rules of bounded-calcs 0.2.0, where the model converted units itself (M7 step 1, and
# the "legacy" arm of the unit-trap set).
TOOL_RULES_LEGACY = """\
2. Read the note and send to `evaluate` only what it actually documents: present, absent, a
   level, or a number. Leave out (or send "unknown") anything the note does not mention.
   Never fill an unmentioned item as absent or normal. If the note says only that a numeric
   item is normal ("vital signs normal"), send "normal" when the schema gives that item a
   normal range, and leave it unknown otherwise. If it gives only bounds ("heart rate over
   100", "SBP in the 90s"), send {"min": x, "max": y}, leaving out the bound it does not
   give. Never invent a number. Convert units yourself to the unit the schema states.
3. If the result is settled, report its category. If not, ask the clinician about the
   decisive unknowns (in the order given), add the answers, and call evaluate again.
4. Items of kind "judgment" (for example HEART history, Wells "alternative diagnosis less
   likely") must come from the clinician, never from your reading of the note."""


LEGACY_INPUT_RULES = (
    "Pass only what is documented or known. Omit an item, or send 'unknown', when the "
    "source does not mention it: never fill it in as absent or normal. A numeric item "
    "the source describes only as normal takes 'normal' if the item has a normal range, "
    "and is unknown otherwise; one the source gives only as bounds ('heart rate over "
    '100\') takes a range {"min": x, "max": y}. Never invent a number. Items of kind '
    "'judgment' must come from the clinician, not from a note."
)


def initial_prompt(
    note: str, tool_calc: bounded_calcs.Calculator, schema: dict[str, Any], rules: str = TOOL_RULES
) -> str:
    cats = ", ".join(c.name for c in tool_calc.categories)
    return f"""\
Calculator: {tool_calc.name} (id: {tool_calc.id})
Decision categories: {cats}

Clinical note:
<note>
{note}
</note>

You work with the bounded-calcs tool, which never assumes an unmentioned finding is normal.
Its rules:
{rules}

Schema of this calculator, as the tool returns it:
{json.dumps(schema)}

Actions (reply with one JSON object per turn):
1. {{"action": "evaluate", "findings": {{"<item id>": <value>, ...}}}}
   Calls the tool with these findings and returns its result.
2. {{"action": "ask", "item": "<item id>", "question": "<your question to the clinician>"}}
   The clinician answers from the patient (or says the value is not available).
3. {{"action": "answer", "category": "<one of: {cats}, {ABSTAIN}>"}}
   Your final decision category. Use "{ABSTAIN}" only if it truly cannot be determined."""


def describe_answer(calc: Calculator, answer: Answer) -> str:
    """The clinician's answer in the tool's vocabulary."""
    pid = answer.param
    p = calc.param(pid)
    if answer.status == "not_available":
        return f"Clinician: {pid} is not available."
    if answer.status == "range":
        unit = p.domain.unit if isinstance(p.domain, NumericDomain) else ""
        return f"Clinician: {pid} is between {fmt(answer.lo)} and {fmt(answer.hi)} {unit}."  # type: ignore[arg-type]
    value = answer.value
    match p.domain:
        case BoolDomain():
            text = "present" if value else "absent"
        case OrdinalDomain(levels=levels):
            text = levels[int(value)]  # type: ignore[arg-type]
        case NumericDomain(unit=unit):
            text = f"{fmt(value)} {unit}"  # type: ignore[arg-type]
    return f"Clinician: {pid} = {text}."


def in_canonical_unit(pid: ParamId, value: Any, site_unit: str | None = None) -> Any:
    """A numeric finding as the agent sent it, converted by this repo's own unit code (not by
    the tool under test) to the item's canonical unit: {"value": x, "unit": u}, bounds with a
    unit, or a plain number or bounds read in the site's unit. Returned unchanged when there
    is nothing to convert or the unit is not one we know (the audit then judges it as sent)."""
    unit = value.get("unit") if isinstance(value, dict) else None
    if isinstance(unit, str) and unit.strip().lower() == bounded_calcs.engine.UNIT_NOT_STATED:
        unit = None
    if unit is None and not isinstance(value, str):
        unit = site_unit
    if not isinstance(unit, str):
        return value
    body = {k: v for k, v in value.items() if k != "unit"} if isinstance(value, dict) else value

    def conv(x: Any) -> Any:
        if isinstance(x, bool) or not isinstance(x, int | float):
            return x
        return to_canonical(pid, float(x), unit)

    try:
        if isinstance(body, dict):
            if set(body) == {"value"}:
                return conv(body["value"])
            return {k: conv(v) for k, v in body.items()}
        return conv(body)
    except UnitError:
        return value


def _as_range(value: Any, normal: tuple[float, float] | None) -> tuple[float, float] | None:
    """A numeric finding as an interval: a number, "normal" (the range in use) or bounds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value), float(value)
    if value == "normal":
        return normal
    if isinstance(value, dict):
        lo, hi = value.get("min", -math.inf), value.get("max", math.inf)
        if isinstance(lo, int | float) and isinstance(hi, int | float):
            return float(lo), float(hi)
    return None


def audit_finding(
    calc: Calculator,
    pid: ParamId,
    value: Any,
    reference: Constraint | None,
    normal: tuple[float, float] | None,
    medication_only: bool,
) -> str:
    """ "ok", "unsupported" (asserted, but neither the note nor the clinician supports it) or
    "wrong" (contradicts the note or the clinician). `reference` is what is known about the
    item: an exact value, an interval (stated normal, or a vague answer), or None."""
    domain = calc.param(pid).domain
    if medication_only and reference is None:
        # A comorbidity conveyed only by a medication: present or unknown are both acceptable.
        return "ok" if value == "present" else "wrong"
    if reference is None:
        return "unsupported"
    match domain:
        case BoolDomain():
            assert isinstance(reference, Exact)
            return "ok" if value == ("present" if reference.value else "absent") else "wrong"
        case OrdinalDomain(levels=levels):
            assert isinstance(reference, Exact)
            return "ok" if value == levels[int(reference.value)] else "wrong"
        case NumericDomain():
            got = _as_range(value, normal)
            if got is None:
                return "wrong"
            if isinstance(reference, Exact):
                v = float(reference.value)
                if got[0] == got[1]:  # a number: must be the documented one (2% for conversions)
                    return "ok" if math.isclose(got[0], v, rel_tol=0.02, abs_tol=0.11) else "wrong"
                return "ok" if got[0] <= v <= got[1] else "wrong"
            # Only an interval is known: a single number would be invented.
            if got[0] == got[1]:
                return "wrong"
            return "ok" if got[0] <= reference.hi and reference.lo <= got[1] else "wrong"
    raise TypeError(domain)


class ToolAgentPolicy:
    id = "s5_tool_agent"

    def __init__(
        self,
        llm: LLM,
        provider: str,
        request_params: dict[str, Any],
        model: str,
        kind: str,
        *,
        normal_ranges: dict[str, tuple[float, float]] | None = None,
        unit_mode: Literal["contract", "legacy"] = "contract",
        site_units: dict[str, str] | None = None,
        max_turns: int = 12,
        max_invalid: int = 3,
    ) -> None:
        self.llm = llm
        self.provider = provider
        self.params = request_params
        self.model = model
        self.kind = kind
        self.normal_ranges = {k: list(v) for k, v in (normal_ranges or {}).items()} or None
        # "contract": as the MCP server works, a number without a unit is refused for an item
        # with several units unless the site has set one. "legacy": bounded-calcs 0.2.0, the
        # model converts units itself and the tool reads every number in the item's unit.
        self.unit_mode = unit_mode
        self.site_units = dict(site_units or {})
        self.max_turns = max_turns
        self.max_invalid = max_invalid
        self.tools = bounded_calcs.load_builtin()

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace:
        tool_calc = self.tools[calc.id]
        contract = self.unit_mode == "contract"
        unit_args: dict[str, Any] = (
            {"units": self.site_units, "require_units": True} if contract else {}
        )
        schema = bounded_calcs.schema(tool_calc, normal_ranges=self.normal_ranges, **unit_args)
        if not contract:  # the 0.2.0 schema said nothing about other units
            for entry in schema["items"]:
                entry.pop("other_units", None)
                entry.pop("unit_labels", None)
            schema["input_rules"] = LEGACY_INPUT_RULES
        normals = {
            i["id"]: (float(i["normal"][0]), float(i["normal"][1]))
            for i in schema["items"]
            if i.get("normal")
        }
        medication_only = {
            t.param for t in case.traps if t.kind == TrapKind.COMORBIDITY_VIA_MEDICATION
        }
        ids = {p.id for p in calc.parameters}
        known = reference_known(case, calc)  # what is truly known; the agent never sees it
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": initial_prompt(
                    note, tool_calc, schema, TOOL_RULES if contract else TOOL_RULES_LEGACY
                ),
            }
        ]
        steps: list[Step] = []
        usage = Usage()
        audit = ToolAudit()
        category: str | None = None
        invalid = 0

        def invalid_turn() -> bool:
            nonlocal invalid
            invalid += 1
            messages.append(
                {"role": "user", "content": "Invalid action; reply with one JSON action."}
            )
            return invalid >= self.max_invalid

        for _ in range(self.max_turns):
            req = LLMRequest(
                provider=self.kind,
                model=self.model,
                system=TOOL_SYSTEM,
                messages=messages,
                params=self.params,
            )
            resp = self.llm.complete(req, provider=self.provider, stage="agent:s5")
            usage.input_tokens += resp.usage.input_tokens
            usage.output_tokens += resp.usage.output_tokens
            usage.cost_usd += resp.usage.cost_usd
            usage.cache_write_tokens += resp.usage.cache_write_tokens
            usage.cache_read_tokens += resp.usage.cache_read_tokens
            usage.latency_s += resp.usage.latency_s
            messages.append({"role": "assistant", "content": resp.text})
            try:
                action = json.loads(resp.text[resp.text.find("{") : resp.text.rfind("}") + 1])
                kind = action["action"]
            except (json.JSONDecodeError, KeyError, TypeError):
                if invalid_turn():
                    break
                continue
            if kind == "answer":
                cat = action.get("category")
                category = cat if cat in calc.category_names() else None
                break
            if kind == "ask" and action.get("item") in ids:
                pid = action["item"]
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
                elif answer.status == "range":
                    known[pid] = Interval(lo=answer.lo, hi=answer.hi)  # type: ignore[arg-type]
                messages.append({"role": "user", "content": describe_answer(calc, answer)})
                continue
            if kind == "evaluate" and isinstance(action.get("findings"), dict):
                findings = action["findings"]
                audit.calls += 1
                for pid, value in findings.items():
                    if pid not in ids or value is None or value == "unknown":
                        continue
                    if isinstance(calc.param(pid).domain, NumericDomain):
                        value = in_canonical_unit(
                            pid, value, self.site_units.get(pid) if contract else None
                        )
                    verdict = audit_finding(
                        calc, pid, value, known.get(pid), normals.get(pid), pid in medication_only
                    )
                    target = {"unsupported": audit.unsupported, "wrong": audit.wrong}.get(verdict)
                    if target is not None and pid not in target:
                        target.append(pid)
                try:
                    result = bounded_calcs.evaluate(
                        tool_calc, findings, normal_ranges=self.normal_ranges, **unit_args
                    )
                except bounded_calcs.InputError as e:
                    audit.tool_errors += 1
                    reply = f"Tool error: {e}"
                else:
                    audit.last_settled = result["settled"]
                    audit.last_category = result["category"]
                    audit.last_possible = result["possible_categories"]
                    reply = "Tool result: " + json.dumps(result)
                messages.append({"role": "user", "content": reply})
                continue
            if invalid_turn():
                break
        final = score_bounds(calc, known)
        audit.answered_against_tool = category is not None and not (
            audit.last_settled and audit.last_category == category
        )
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
            tool_audit=audit,
        )
