"""M7: a unit-trap set for the bounded-calcs tool (S5), on templated notes.

One question: when a laboratory value can be written in several units, does the agent get the
value into the tool correctly? CURB-65 urea is the item (mmol/L, or mg/dL as urea or as urea
nitrogen, BUN). The cases are a full grid, not a sample:

  value         nine BUN levels from 11 to 42 mg/dL, on both sides of the CURB-65 cut-off
                (urea > 7 mmol/L, which is BUN > 19.6 mg/dL)
  presentation  how the note writes it: in mmol/L; as BUN in mg/dL; as urea in mg/dL; as BUN
                with no unit; as urea with no unit
  background    the other items give 1 point (urea decides low or moderate) or 2 points (urea
                decides moderate or high)

The notes are one fixed template with the values filled in (not rendered by a model), so the
only thing that varies is the urea line. The truth is the value as written, converted by
`units.py`. In the two presentations without a unit the note does not determine urea
(`not_documented`): the right move is to ask, and the simulated clinician answers in mmol/L.

Three ways of working with the tool:
  legacy    bounded-calcs 0.2.0: the model converts units itself, every number is read in mmol/L
  contract  the model sends the unit the note gives; a number without a unit is refused
  site_bun  contract, at a site that has declared it reports BUN in mg/dL

Outputs under `<run>/tooleval/unit_traps/`: `<mode>__<provider>.jsonl` traces, `cases.csv`,
`summary.csv` and `summary.md`.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.config import RunConfig
from calc_bounds.eval.paper import to_markdown
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.pipeline import calculators, make_clinician, make_llm, run_dir
from calc_bounds.policies import Trace
from calc_bounds.policies.tool_agent import ToolAgentPolicy
from calc_bounds.types import DocumentedState
from calc_bounds.units import to_canonical

type Mode = Literal["legacy", "contract", "site_bun"]
MODES: tuple[Mode, ...] = ("legacy", "contract", "site_bun")

BUN_LEVELS = [11, 14, 17, 19, 20, 22, 25, 31, 42]  # mg/dL; the cut-off is 19.6
UREA_PER_BUN = 6.006 / 2.8014  # mg/dL of urea per mg/dL of urea nitrogen (factors of units.py)

# name -> (unit the number is in, what the note writes, whether the note states the unit)
PRESENTATIONS: dict[str, tuple[str, str, bool]] = {
    "mmol": ("mmol/L", "Urea {v} mmol/L", True),
    "bun": ("bun_mg/dL", "BUN {v} mg/dL", True),
    "urea_mgdl": ("urea_mg/dL", "Urea {v} mg/dL", True),
    "bun_no_unit": ("bun_mg/dL", "BUN {v}", False),
    "urea_no_unit": ("urea_mg/dL", "Urea {v}", False),
}

# The other CURB-65 items: age 65 or more is 1 point; respiratory rate 30 or more is another.
BACKGROUNDS: dict[str, dict[str, float | bool]] = {
    "one_point": {"confusion": False, "resp_rate": 18.0, "sbp": 124.0, "dbp": 76.0, "age": 70.0},
    "two_points": {"confusion": False, "resp_rate": 32.0, "sbp": 118.0, "dbp": 70.0, "age": 72.0},
}

NOTE = """\
Chief Complaint: Cough and fever.

HPI: {age}-year-old with four days of productive cough, fever and worsening shortness of \
breath. No chest pain.

Exam:
Vitals: T 38.6 C, HR 98, RR {rr}, BP {sbp}/{dbp}, SpO2 93% on room air.
General: Alert and fully oriented, not confused.
Pulm: Crackles at the right base.

Results: Chest x-ray shows a right lower lobe infiltrate. WBC 14.2. {urea}. Creatinine normal.

Assessment/Plan: Community-acquired pneumonia. Antibiotics started."""


def shown_value(presentation: str, bun: int) -> float:
    """The number the note prints for a BUN level in this presentation."""
    unit = PRESENTATIONS[presentation][0]
    if unit == "bun_mg/dL":
        return float(bun)
    if unit == "urea_mg/dL":
        return float(round(bun * UREA_PER_BUN))
    return round(to_canonical("urea", bun, "bun_mg/dL"), 1)


def build_cases(cfg: RunConfig) -> tuple[list[PatientCase], dict[str, str]]:
    """The grid of cases and their templated notes."""
    calc = calculators(cfg)["curb65"]
    cases: list[PatientCase] = []
    notes: dict[str, str] = {}
    for background, rest in BACKGROUNDS.items():
        for presentation, (unit, phrase, labelled) in PRESENTATIONS.items():
            for bun in BUN_LEVELS:
                shown = shown_value(presentation, bun)
                truth = {**rest, "urea": to_canonical("urea", shown, unit)}
                score, category = calc.evaluate(truth)
                documented = {
                    pid: DocumentedState.NEGATIVE
                    if pid == "confusion"
                    else DocumentedState.POSITIVE
                    for pid in truth
                }
                if not labelled:  # a number with no unit does not determine the item
                    documented["urea"] = DocumentedState.NOT_DOCUMENTED
                case_id = f"units-{background}-{presentation}-{bun}"
                cases.append(
                    PatientCase(
                        case_id=case_id,
                        seed=0,
                        calculator="curb65",
                        truth=truth,
                        documented=documented,
                        true_score=score,
                        true_category=category,
                        determined_from_note=labelled,
                    )
                )
                notes[case_id] = NOTE.format(
                    age=int(rest["age"]),
                    rr=int(rest["resp_rate"]),
                    sbp=int(rest["sbp"]),
                    dbp=int(rest["dbp"]),
                    urea=phrase.format(v=f"{shown:g}"),
                )
    return cases, notes


def out_dir(cfg: RunConfig) -> Path:
    return run_dir(cfg) / "tooleval" / "unit_traps"


def run(cfg: RunConfig, provider: str, mode: Mode) -> Path:
    te = cfg.tool_eval
    assert te is not None, "config has no `tool_eval` section"
    assert provider in te.providers, f"{provider!r} is not listed in tool_eval.providers"
    pcfg = cfg.providers[provider]
    llm = make_llm(cfg)
    policy = ToolAgentPolicy(
        llm,
        provider,
        pcfg.request_params(),
        pcfg.model,
        pcfg.kind,
        normal_ranges=te.normal_ranges,
        unit_mode="legacy" if mode == "legacy" else "contract",
        site_units={"urea": "bun_mg/dL"} if mode == "site_bun" else None,
        max_turns=te.max_turns,
    )
    cases, notes = build_cases(cfg)
    calc = calculators(cfg)["curb65"]

    def run_one(case: PatientCase) -> Trace:
        return policy.run(case, notes[case.case_id], calc, None, make_clinician(cfg, case, calc))  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=te.max_workers) as pool:
        traces = list(pool.map(run_one, cases))
    path = out_dir(cfg) / f"{mode}__{provider}.jsonl"
    write_jsonl(path, traces)
    print(f"{mode} / {provider}: {len(traces)} cases; newly spent ${llm.spent_usd:.2f}")
    return path


def case_table(cfg: RunConfig) -> pd.DataFrame:
    """One row per (mode, provider, case) over every trace file present."""
    cases = {c.case_id: c for c in build_cases(cfg)[0]}
    rows = []
    for path in sorted(out_dir(cfg).glob("*.jsonl")):
        mode, provider = path.stem.split("__")
        for t in read_jsonl(path, Trace):
            case = cases[t.case_id]
            _, background, presentation, bun = t.case_id.split("-")
            audit = t.tool_audit
            assert audit is not None
            rows.append(
                {
                    "mode": mode,
                    "provider": provider,
                    "presentation": presentation,
                    "background": background,
                    "bun": int(bun),
                    "case_id": t.case_id,
                    "true_category": case.true_category,
                    "final_category": t.final_category,
                    "correct": t.final_category == case.true_category,
                    "under_triage": t.final_category is not None
                    and ["low", "moderate", "high"].index(t.final_category)
                    < ["low", "moderate", "high"].index(case.true_category),
                    "wrong_urea": "urea" in audit.wrong,
                    "unsupported_urea": "urea" in audit.unsupported,
                    "asked_urea": any(s.question == "urea" for s in t.steps),
                    "tool_errors": audit.tool_errors,
                    "cost_usd": t.usage.cost_usd if t.usage else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["provider", "mode", "presentation"], sort=False)
        .agg(
            n=("correct", "size"),
            correct=("correct", "mean"),
            under_triage=("under_triage", "mean"),
            wrong_urea=("wrong_urea", "mean"),
            unsupported_urea=("unsupported_urea", "mean"),
            asked_urea=("asked_urea", "mean"),
            tool_errors_per_case=("tool_errors", "mean"),
            cost_usd_per_case=("cost_usd", "mean"),
        )
        .reset_index()
    )


def report(cfg: RunConfig) -> Path:
    table = case_table(cfg)
    out = out_dir(cfg)
    table.to_csv(out / "cases.csv", index=False)
    summary = summarize(table)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary.round(3)) + "\n", encoding="utf-8")
    return out / "summary.md"
