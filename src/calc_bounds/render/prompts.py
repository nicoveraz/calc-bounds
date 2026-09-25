"""Render and judge prompts."""

from importlib import resources

from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.render import vocab
from calc_bounds.render.facts import build_facts
from calc_bounds.types import NumericDomain, OrdinalDomain

PROMPT_VERSION = "v3"
"""Bump whenever render prompt wording changes; recorded on every note."""

LOCALE_NAMES = {"en-US": "US English", "es-CL": "Chilean Spanish"}

RENDER_SYSTEM = """\
You write realistic clinical notes for a research dataset. Every patient is fictional; the \
notes are synthetic. You follow the fact sheet exactly: each listed item must be documented \
exactly as instructed, and items marked DO NOT MENTION must be completely absent and not \
inferable. Output only the note text."""


def style_guide(locale: str) -> str:
    return resources.files("calc_bounds.render.styles").joinpath(f"{locale}.md").read_text()


def render_prompt(case: PatientCase, calc: Calculator, locale: str) -> str:
    facts = build_facts(case, calc)
    lines = "\n".join(f"- [{f.param}] {f.instruction}" for f in facts)
    return f"""\
Write one clinical note in {LOCALE_NAMES[locale]} ({locale}).

Setting: {vocab.SETTINGS[calc.id]}

Style guide:
{style_guide(locale).strip()}

Fact sheet (each item is mandatory):
{lines}

Rules:
- Write numbers exactly as given in the fact sheet (same digits and decimals), with the unit \
given; you may use natural phrasing and abbreviations around them.
- Express each item in natural clinical language, as a clinician would chart it; do not copy \
the fact-sheet wording. Numbers, units and ordinal meanings must stay exactly as specified.
- Every STATE item must appear explicitly in the note, including negative ones (e.g. \
"no diabetes", "never smoked").
- Follow each item exactly. Do not add information about any fact-sheet item beyond what it \
says, and do not include anything that implies an item marked DO NOT MENTION.
- No blanket or complete statements that would cover a DO NOT MENTION item: no "Medications: \
none", "PMH: none/unremarkable", "ROS otherwise negative", "vitals stable" or "labs normal". \
If a list (medications, history) would normally reveal such an item, leave that list out.
- You may add other realistic details (presenting complaint, unrelated history, unrelated \
examination findings, investigations and plan) as long as they do not imply any fact-sheet \
item.
- Do not name any clinical score or rule ({", ".join(vocab.FORBIDDEN_TERMS)}), do not compute \
scores, and do not state a risk category or disposition that reveals one.
- Do not refer to these instructions, the fact sheet, or the note being synthetic.
- Output only the note, 150-350 words, plain text (simple section headings allowed)."""


JUDGE_SYSTEM = """\
You audit synthetic clinical notes against a list of parameters. For each parameter, decide \
only from the note text how it is documented. Be strict: a parameter that is not stated but \
can reasonably be inferred (e.g. from a medication, a related finding, or a blanket statement \
such as 'vitals normal') is 'implied'. Output only JSON."""


def _describe(calc: Calculator) -> str:
    out = []
    for p in calc.parameters:
        match p.domain:
            case NumericDomain(unit=unit):
                kind = f"numeric ({unit})"
            case OrdinalDomain(levels=levels):
                kind = "one of: " + ", ".join(levels)
            case _:
                kind = "yes/no finding"
        out.append(f"- {p.id}: {p.label} [{kind}]")
    return "\n".join(out)


def judge_prompt(note: str, calc: Calculator) -> str:
    return f"""\
Note:
<note>
{note}
</note>

Parameters:
{_describe(calc)}

For every parameter, return one object:
{{"param": "<id>", "status": "stated" | "negated_or_normal" | "not_mentioned" | "implied", \
"level": "<for 'one of' parameters: the level stated, else null>", \
"quote": "<shortest exact quote from the note supporting the status, or null>"}}

- stated: the value or finding is explicitly documented (for yes/no findings: present).
- negated_or_normal: explicitly absent, denied, or documented as normal without a value.
- not_mentioned: nothing in the note documents or implies it.
- implied: not explicitly documented, but inferable from the note.

Return a JSON array with exactly one object per parameter, in the order listed, and nothing else."""
