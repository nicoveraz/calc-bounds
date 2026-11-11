"""M7: run LLM agents on the synthetic notes with the bounded-calcs tool (S5) or with the plain
calculator (S2), per provider, and tabulate accuracy and safety.

Outputs under `<run>/tooleval/`: `traces/<arm>__<provider>__<render>[__<clinician>].jsonl`,
and `summary.csv` / `summary.md` over every trace file present.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import bounded_calcs
import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.config import RunConfig
from calc_bounds.eval import metrics
from calc_bounds.eval.paper import to_markdown
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.pipeline import (
    calculators,
    make_clinician,
    make_llm,
    notes_path,
    run_dir,
    select_cases,
)
from calc_bounds.policies import Trace
from calc_bounds.policies.llm_agent import LLMAgentPolicy
from calc_bounds.policies.tool_agent import ToolAgentPolicy
from calc_bounds.render import RenderedNote

type Arm = Literal["tool", "calculator"]


def out_dir(cfg: RunConfig) -> Path:
    return run_dir(cfg) / "tooleval"


def traces_path(cfg: RunConfig, arm: Arm, provider: str) -> Path:
    assert cfg.tool_eval is not None
    label = f"{arm}__{provider}__{cfg.tool_eval.render}"
    if cfg.simulator.name != "ideal":
        label += f"__{cfg.simulator.name}"
    return out_dir(cfg) / "traces" / f"{label}.jsonl"


def eval_cases(cfg: RunConfig, per_calc: int | None) -> tuple[list[PatientCase], dict[str, str]]:
    """Cases of the calculators bounded-calcs implements, with a note that passed validation.
    `per_calc` takes the seeded, nested subset used for pilots."""
    assert cfg.tool_eval is not None
    cases = select_cases(cfg, read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase), per_calc)
    notes = {
        n.case_id: n.text
        for n in read_jsonl(notes_path(cfg, cfg.tool_eval.render), RenderedNote)
        if not n.validation_failed
    }
    supported = set(bounded_calcs.load_builtin())
    return [c for c in cases if c.calculator in supported and c.case_id in notes], notes


def run(cfg: RunConfig, arm: Arm, provider: str, per_calc: int | None = None) -> Path:
    te = cfg.tool_eval
    assert te is not None, "config has no `tool_eval` section"
    assert provider in te.providers, f"{provider!r} is not listed in tool_eval.providers"
    pcfg = cfg.providers[provider]
    llm = make_llm(cfg)
    policy: ToolAgentPolicy | LLMAgentPolicy
    if arm == "tool":
        policy = ToolAgentPolicy(
            llm,
            provider,
            pcfg.request_params(),
            pcfg.model,
            pcfg.kind,
            normal_ranges=te.normal_ranges,
            max_turns=te.max_turns,
        )
    else:
        policy = LLMAgentPolicy(
            llm, provider, pcfg.request_params(), pcfg.model, pcfg.kind, max_turns=te.max_turns
        )
    cases, notes = eval_cases(cfg, per_calc)
    calcs = calculators(cfg)

    def run_one(case: PatientCase) -> Trace:
        calc = calcs[case.calculator]
        return policy.run(case, notes[case.case_id], calc, None, make_clinician(cfg, case, calc))  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=te.max_workers) as pool:
        traces = list(pool.map(run_one, cases))
    path = traces_path(cfg, arm, provider)
    write_jsonl(path, traces)
    usage = [t.usage for t in traces if t.usage]
    recorded = sum(u.cost_usd for u in usage)
    print(
        f"{arm} / {provider}: {len(traces)} cases; tokens: "
        f"{sum(u.input_tokens for u in usage)} input, "
        f"{sum(u.cache_write_tokens for u in usage)} cache write, "
        f"{sum(u.cache_read_tokens for u in usage)} cache read, "
        f"{sum(u.output_tokens for u in usage)} output; "
        f"cost of these traces ${recorded:.2f} (newly spent this run ${llm.spent_usd:.2f})"
    )
    return path


def case_table(cfg: RunConfig) -> pd.DataFrame:
    """One row per (arm, provider, clinician, case) over every trace file present."""
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    frames = []
    for path in sorted((out_dir(cfg) / "traces").glob("*.jsonl")):
        arm, provider, render, *rest = path.stem.split("__")
        traces = read_jsonl(path, Trace)
        table = metrics.case_table(cases, traces)
        table.insert(0, "clinician", rest[0] if rest else "ideal")
        table.insert(0, "render", render)
        table.insert(0, "provider", provider)
        table.insert(0, "arm", arm)
        audits = [t.tool_audit for t in traces]
        table["tool_calls"] = [a.calls if a else None for a in audits]
        table["silent_fill"] = [bool(a.unsupported) if a else None for a in audits]
        table["n_silent_fill_items"] = [len(a.unsupported) if a else None for a in audits]
        table["wrong_finding"] = [bool(a.wrong) if a else None for a in audits]
        table["answered_against_tool"] = [a.answered_against_tool if a else None for a in audits]
        table["cost_usd"] = [t.usage.cost_usd if t.usage else 0.0 for t in traces]
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby(["arm", "provider", "render", "clinician"])
    committed = table[~table["abstained"]].groupby(["arm", "provider", "render", "clinician"])
    out = pd.DataFrame(
        {
            "n_cases": g.size(),
            "accuracy": g["correct"].mean(),
            "abstain_rate": g["abstained"].mean(),
            "accuracy_when_committed": committed["correct"].mean(),
            "under_triage_rate": g["under_triage"].mean(),
            "over_triage_rate": g["over_triage"].mean(),
            "premature_commitment_rate": g["premature_commitment"].mean(),
            "mean_questions": g["n_questions"].mean(),
            "irrelevant_question_rate": g["n_irrelevant_questions"].sum()
            / g["n_questions"].sum().where(lambda s: s > 0),
            "silent_fill_rate": g["silent_fill"].apply(lambda s: s.dropna().astype(float).mean()),
            "wrong_finding_rate": g["wrong_finding"].apply(
                lambda s: s.dropna().astype(float).mean()
            ),
            "answered_against_tool_rate": g["answered_against_tool"].apply(
                lambda s: s.dropna().astype(float).mean()
            ),
            "cost_usd_per_case": g["cost_usd"].mean(),
        }
    )
    return out.reset_index()


def report(cfg: RunConfig) -> Path:
    table = case_table(cfg)
    out = out_dir(cfg)
    table.to_csv(out / "cases.csv", index=False)
    summary = summarize(table)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary.round(3)) + "\n")
    return out / "summary.md"
