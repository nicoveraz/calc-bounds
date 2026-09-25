"""Fact sheet: the per-parameter instructions a note must follow, plus the strings a faithful
note must contain (used by the deterministic validation pass)."""

from pydantic import BaseModel

from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase, Trap, TrapKind
from calc_bounds.render import vocab
from calc_bounds.types import (
    BoolDomain,
    DocumentedState,
    NumericDomain,
    OrdinalDomain,
    ParamId,
    Value,
)


def fmt(v: Value) -> str:
    """Surface form of a number exactly as the note must show it."""
    if isinstance(v, bool):
        return str(v)
    x = float(v)
    return str(int(x)) if x.is_integer() else f"{x:g}"


class Fact(BaseModel):
    param: ParamId
    state: DocumentedState
    instruction: str
    required_numbers: list[str] = []
    """Numbers (surface form) that must appear in the note."""
    leak_numbers: list[str] = []
    """Numbers whose presence may indicate a not-documented value leaked into the note."""


def _trap_for(case: PatientCase, pid: ParamId) -> Trap | None:
    return next((t for t in case.traps if t.param == pid), None)


def build_facts(case: PatientCase, calc: Calculator) -> list[Fact]:
    facts: list[Fact] = []
    for p in calc.parameters:
        v = case.truth[p.id]
        state = case.documented[p.id]
        trap = _trap_for(case, p.id)
        required: list[str] = []
        leaks: list[str] = []
        if state == DocumentedState.NOT_DOCUMENTED:
            text = (
                f"DO NOT MENTION {p.label}. Include nothing from which it could be inferred "
                "(no value, no 'normal'/'abnormal', no related medication, no blanket "
                "statements such as 'vitals stable' or 'labs unremarkable' that would cover it)."
            )
            if isinstance(p.domain, NumericDomain):
                leaks.append(fmt(v))
        else:
            match p.domain:
                case NumericDomain():
                    if state == DocumentedState.NEGATIVE:
                        text = (
                            f"STATE AS NORMAL, WITHOUT A NUMBER: "
                            f"{vocab.NUMERIC_NORMAL.get(p.id, p.label + ' normal')}."
                        )
                    elif trap is not None and trap.kind == TrapKind.MIXED_UNITS:
                        shown = fmt(trap.detail["value_in_unit"])
                        phrase = vocab.MIXED_UNIT_PHRASES[str(trap.detail["unit"])]
                        text = f"STATE (number and unit exactly): {phrase.format(v=shown)}."
                        required.append(shown)
                    else:
                        phrase = vocab.NUMERIC_PHRASES[p.id]
                        text = f"STATE (number exactly): {phrase.format(v=fmt(v))}."
                        required.append(fmt(v))
                case OrdinalDomain():
                    if state == DocumentedState.NEGATIVE:
                        text = f"STATE AS NORMAL/NEGATIVE: {vocab.LEVELS[p.id][0]}."
                    else:
                        text = f"STATE: {vocab.LEVELS[p.id][int(v)]}."
                case BoolDomain():
                    pos, neg = vocab.BOOL_PHRASES.get(
                        p.id,
                        (f"the patient has: {p.label}", f"the patient does not have: {p.label}"),
                    )
                    text = f"STATE: {pos if state == DocumentedState.POSITIVE else neg}."
        if trap is not None and trap.kind == TrapKind.COMORBIDITY_VIA_MEDICATION:
            text = "STATE ONLY INDIRECTLY: " + _trap_instruction(trap, p.label, v)
        elif trap is not None:
            text += " " + _trap_instruction(trap, p.label, v)
            for key in ("prior_encounter_value", "distractor_value"):
                if key in trap.detail:
                    required.append(fmt(trap.detail[key]))  # type: ignore[arg-type]
        facts.append(
            Fact(
                param=p.id,
                state=state,
                instruction=text,
                required_numbers=required,
                leak_numbers=leaks,
            )
        )
    return facts


def _trap_instruction(trap: Trap, label: str, v: Value) -> str:
    match trap.kind:
        case TrapKind.NEGATION:
            return (
                "Write this negation inside a sentence or list that also contains positive "
                "findings (e.g. 'no X, but reports Y'), so it is easy to misread."
            )
        case TrapKind.MIXED_UNITS:
            return "Use only this unit for this value."
        case TrapKind.MULTIPLE_ENCOUNTERS:
            prior = fmt(trap.detail["prior_encounter_value"])  # type: ignore[arg-type]
            return (
                f"Also mention a {label} of {prior} from a previous visit (clearly dated, "
                "e.g. '3 months ago'); today's value above is the current one."
            )
        case TrapKind.CONTRADICTORY_VALUES:
            d = fmt(trap.detail["distractor_value"])  # type: ignore[arg-type]
            return (
                f"Also mention an earlier reading today of {d} for {label} (e.g. at triage), "
                f"clearly superseded by the repeat value {fmt(v)} stated above."
            )
        case TrapKind.COMORBIDITY_VIA_MEDICATION:
            cond = vocab.MEDICATION_CONDITION[trap.param]
            med = trap.detail["medication"]
            return (
                f"Do NOT name {cond} anywhere; convey it only by listing {med} among the home "
                "medications."
            )
