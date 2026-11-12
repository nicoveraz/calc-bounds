"""M7: run LLM agents on the synthetic notes with the bounded-calcs tool (S5) or with the plain
calculator (S2), per provider, and tabulate accuracy and safety.

Outputs under `<run>/tooleval/`: `traces/<arm>__<provider>__<render>[__<clinician>].jsonl`,
and `summary.csv` / `summary.md` over every trace file present.
"""

import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import bounded_calcs
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

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


@functools.cache
def _tools() -> dict[str, bounded_calcs.Calculator]:
    return bounded_calcs.load_builtin()


def judgment_confirmations(trace: Trace) -> int:
    """Questions about a clinician-judgment item that were not decision-relevant by the
    reference knowledge: the note already states the judgment, and the agent asked the
    clinician anyway. The tool's rules require that, so these are counted apart from
    questions that could not change the category for any other reason."""
    items = _tools()[trace.calculator].items
    return sum(
        s.question is not None
        and s.relevant is not None
        and s.question not in s.relevant
        and items[s.question].kind == "judgment"
        for s in trace.steps
    )


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
        table["n_judgment_confirmations"] = [judgment_confirmations(t) for t in traces]
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
            "judgment_confirmations_per_case": g["n_judgment_confirmations"].mean(),
            "other_irrelevant_question_rate": (
                g["n_irrelevant_questions"].sum() - g["n_judgment_confirmations"].sum()
            )
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


BINARY_OUTCOMES = ("correct", "abstained", "under_triage", "over_triage", "premature_commitment")


def _paired(x: pd.DataFrame, y: pd.DataFrame) -> list[dict[str, object]]:
    """Paired tests between two sets of rows on the cases they share: an exact McNemar test
    per yes/no outcome, a Wilcoxon signed-rank test on the number of questions. Two-sided."""
    x, y = x.set_index("case_id"), y.set_index("case_id")
    ids = x.index.intersection(y.index)
    x, y = x.loc[ids], y.loc[ids]
    rows: list[dict[str, object]] = []
    for outcome in BINARY_OUTCOMES:
        a, b = x[outcome].astype(bool), y[outcome].astype(bool)
        only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
        table = [[int((a & b).sum()), only_a], [only_b, int((~a & ~b).sum())]]
        rows.append(
            {
                "outcome": outcome,
                "n": len(ids),
                "a": float(a.mean()),
                "b": float(b.mean()),
                "only_a": only_a,
                "only_b": only_b,
                "test": "mcnemar_exact",
                "p": float(mcnemar(table, exact=True).pvalue),
            }
        )
    qa, qb = x["n_questions"], y["n_questions"]
    differs = bool((qa != qb).any())
    rows.append(
        {
            "outcome": "n_questions",
            "n": len(ids),
            "a": float(qa.mean()),
            "b": float(qb.mean()),
            "only_a": int((qa > qb).sum()),  # cases where a asked more
            "only_b": int((qb > qa).sum()),
            "test": "wilcoxon",
            "p": float(wilcoxon(qa, qb).pvalue) if differs else 1.0,
        }
    )
    return rows


def comparisons(table: pd.DataFrame) -> pd.DataFrame:
    """Paired tests on the same cases: the tool arm against the calculator arm for each model
    clinician and set of notes, and one model against the other within each arm. `p` is not
    corrected for the number of tests; `p_holm` is, over every row of the table."""
    rows: list[dict[str, object]] = []

    def add(what: str, fixed: dict[str, str], key: str, a: str, b: str) -> None:
        sub = table
        for column, value in fixed.items():
            sub = sub[sub[column] == value]
        x, y = sub[sub[key] == a], sub[sub[key] == b]
        if len(x) and len(y):
            rows.extend(
                {"comparison": what, **fixed, "a_is": a, "b_is": b, **r} for r in _paired(x, y)
            )

    providers = sorted(table["provider"].unique())
    for render in sorted(table["render"].unique()):
        for clinician in sorted(table["clinician"].unique()):
            fixed = {"render": render, "clinician": clinician}
            for provider in providers:
                add(
                    "tool vs calculator",
                    {**fixed, "provider": provider},
                    "arm",
                    "tool",
                    "calculator",
                )
            for arm in sorted(table["arm"].unique()):
                for i, first in enumerate(providers):
                    for second in providers[i + 1 :]:
                        add("model vs model", {**fixed, "arm": arm}, "provider", first, second)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


def report(cfg: RunConfig) -> Path:
    table = case_table(cfg)
    out = out_dir(cfg)
    table.to_csv(out / "cases.csv", index=False)
    summary = summarize(table)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary.round(3)) + "\n")
    tests = comparisons(table)
    tests.to_csv(out / "comparisons.csv", index=False)
    (out / "comparisons.md").write_text(to_markdown(tests.round(4)) + "\n")
    return out / "summary.md"
