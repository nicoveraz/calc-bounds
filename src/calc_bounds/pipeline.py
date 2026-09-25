"""Pipeline stages driven by a RunConfig. Outputs go to <output_dir>/<run_id>/."""

from pathlib import Path

import numpy as np

from calc_bounds.calculators import Calculator, get_calculator
from calc_bounds.cohort import PatientCase, generate_cohort
from calc_bounds.cohort.generate import stable_seed
from calc_bounds.config import ProviderConfig, RunConfig
from calc_bounds.eval import metrics, plots
from calc_bounds.extraction import Extractor
from calc_bounds.extraction.oracle import OracleExtractor
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.llm import LLM, DiskCache, LLMRequest, PendingResponseError, cache_key
from calc_bounds.llm.clients import make_client
from calc_bounds.llm.session import PendingItem, import_responses, write_pending
from calc_bounds.policies import POLICIES, Trace
from calc_bounds.render import JudgeResult, RenderedNote, ValidationIssue
from calc_bounds.render.export import export_review_sample
from calc_bounds.render.prompts import (
    JUDGE_SYSTEM,
    RENDER_SYSTEM,
    judge_prompt,
    render_prompt,
)
from calc_bounds.render.validate import judge_issues, parse_judge, rule_issues
from calc_bounds.simulator import SimulatedClinician, draw_unavailable


def run_dir(cfg: RunConfig) -> Path:
    return cfg.output_dir / cfg.run_id


def calculators(cfg: RunConfig) -> dict[str, Calculator]:
    calcs = {c: get_calculator(c, cfg.calculator_options.get(c)) for c in cfg.calculators}
    known = {p.id for calc in calcs.values() for p in calc.parameters}
    unknown = set(cfg.simulator.unavailable_rate) - known
    if unknown:
        raise ValueError(f"simulator.unavailable_rate: unknown params {sorted(unknown)}")
    return calcs


def make_cohort(cfg: RunConfig) -> Path:
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    cases = generate_cohort(list(calculators(cfg).values()), cfg.cohort, cfg.seed)
    write_jsonl(out / "cohort.jsonl", cases)
    return out / "cohort.jsonl"


def _extractor(cfg: RunConfig, cases: list[PatientCase]) -> Extractor:
    if cfg.extraction.kind == "oracle":
        return OracleExtractor({c.case_id: c for c in cases})
    raise NotImplementedError("LLM extraction arrives in M4")


def run_policies(cfg: RunConfig) -> Path:
    out = run_dir(cfg)
    cases = read_jsonl(out / "cohort.jsonl", PatientCase)
    calcs = calculators(cfg)
    extractor = _extractor(cfg, cases)
    missing = [p for p in cfg.policies if p not in POLICIES]
    if missing:
        raise NotImplementedError(f"policies not implemented yet: {missing}")
    traces: list[Trace] = []
    for case in cases:
        calc = calcs[case.calculator]
        unavailable = draw_unavailable(
            [p.id for p in calc.parameters],
            cfg.simulator.unavailable_rate,
            stable_seed(cfg.seed, case.case_id, "simulator"),
        )
        for pid in cfg.policies:
            clinician = SimulatedClinician(case.truth, unavailable)
            note = ""  # M2: oracle extraction needs no note; rendered notes arrive in M3.
            traces.append(POLICIES[pid].run(case, note, calc, extractor, clinician))
    write_jsonl(out / "traces.jsonl", traces)
    return out / "traces.jsonl"


def evaluate(cfg: RunConfig) -> Path:
    out = run_dir(cfg)
    cases = read_jsonl(out / "cohort.jsonl", PatientCase)
    traces = read_jsonl(out / "traces.jsonl", Trace)
    table = metrics.case_table(cases, traces)
    table.to_csv(out / "cases.csv", index=False)
    overall = metrics.summary(table, ["policy"])
    per_calc = metrics.summary(table, ["policy", "calculator"])
    by_coverage = metrics.summary(table, ["policy", "determined_from_note"])
    overall.to_csv(out / "summary.csv", index=False)
    per_calc.to_csv(out / "summary_by_calculator.csv", index=False)
    by_coverage.to_csv(out / "summary_by_coverage.csv", index=False)
    ext = metrics.extraction_table(cases, traces)
    ext.groupby(["policy", "documented"])["correct"].mean().reset_index().to_csv(
        out / "extraction_by_documented_state.csv", index=False
    )
    plots.accuracy_vs_questions(overall, per_calc, out / "accuracy_vs_questions.png")
    return out


# --- M3: rendering, validation, review --------------------------------------------------------


def make_llm(cfg: RunConfig) -> LLM:
    return LLM(
        lambda name: make_client(cfg.providers[name]),
        DiskCache(cfg.cache_dir),
        max_cost_usd=cfg.max_cost_usd,
        ledger_path=run_dir(cfg) / "usage.jsonl",
    )


def _request(p: ProviderConfig, system: str, user: str) -> LLMRequest:
    return LLMRequest(
        provider=p.kind,
        model=p.model,
        system=system,
        messages=[{"role": "user", "content": user}],
        params=p.request_params(),
    )


def select_cases(
    cfg: RunConfig, cases: list[PatientCase], per_calc: int | None
) -> list[PatientCase]:
    """All cases, or a seeded subset per calculator split evenly by coverage stratum."""
    if per_calc is None:
        return cases
    chosen: list[PatientCase] = []
    for calc_id in cfg.calculators:
        for determined in (True, False):
            pool = [
                c for c in cases if c.calculator == calc_id and c.determined_from_note == determined
            ]
            k = per_calc // 2 + (per_calc % 2 if not determined else 0)
            # A fixed seeded permutation, truncated: subsets of different sizes are nested
            # (pilot within stage 1 within the full set), so cached notes carry over.
            rng = np.random.default_rng(stable_seed(cfg.seed, "subset", calc_id, str(determined)))
            chosen += [pool[i] for i in rng.permutation(len(pool))[:k]]
    order = {c.case_id: i for i, c in enumerate(cases)}
    return sorted(chosen, key=lambda c: order[c.case_id])


def notes_path(cfg: RunConfig, render: str) -> Path:
    return run_dir(cfg) / "notes" / f"{render}.jsonl"


def pending_path(cfg: RunConfig, stage: str, render: str) -> Path:
    return run_dir(cfg) / "pending" / f"{stage}-{render}.jsonl"


def _finish_pending(path: Path, pending: list[PendingItem]) -> None:
    if pending:
        write_pending(path, pending)
    elif path.exists():
        path.unlink()


def render_notes(cfg: RunConfig, render: str) -> dict[str, int]:
    """Render (or load from cache) every selected case x locale. Session-provider cache misses
    are exported to pending/render-<name>.jsonl instead of failing."""
    spec = cfg.renders[render]
    pcfg = cfg.providers[spec.provider]
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    calcs = calculators(cfg)
    llm = make_llm(cfg)
    notes: list[RenderedNote] = []
    pending: list[PendingItem] = []
    for case in select_cases(cfg, cases, spec.subset_per_calculator):
        calc = calcs[case.calculator]
        for locale in spec.locales:
            req = _request(pcfg, RENDER_SYSTEM, render_prompt(case, calc, locale))
            try:
                resp = llm.complete(req, provider=spec.provider, stage=f"render:{render}")
            except PendingResponseError as e:
                pending.append(PendingItem(key=e.key, request=req))
                continue
            notes.append(
                RenderedNote(
                    case_id=case.case_id,
                    render=render,
                    locale=locale,
                    text=resp.text,
                    provider=pcfg.kind,
                    model=pcfg.model,
                    cache_key=cache_key(req),
                    issues=rule_issues(case, calc, resp.text),
                )
            )
    write_jsonl(notes_path(cfg, render), notes)
    _finish_pending(pending_path(cfg, "render", render), pending)
    return {
        "notes": len(notes),
        "pending": len(pending),
        "notes_with_rule_errors": sum(any(i.severity == "error" for i in n.issues) for n in notes),
    }


def judge_notes(cfg: RunConfig, render: str) -> dict[str, int]:
    judge = cfg.validation.judge_provider
    if judge is None:
        raise ValueError("validation.judge_provider is not configured")
    pcfg = cfg.providers[judge]
    if pcfg.model == cfg.providers[cfg.renders[render].provider].model:
        raise ValueError("judge model must differ from the renderer model")
    cases = {c.case_id: c for c in read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)}
    calcs = calculators(cfg)
    llm = make_llm(cfg)
    results: list[JudgeResult] = []
    pending: list[PendingItem] = []
    for note in read_jsonl(notes_path(cfg, render), RenderedNote):
        case = cases[note.case_id]
        calc = calcs[case.calculator]
        req = _request(pcfg, JUDGE_SYSTEM, judge_prompt(note.text, calc))
        try:
            resp = llm.complete(req, provider=judge, stage=f"judge:{render}")
        except PendingResponseError as e:
            pending.append(PendingItem(key=e.key, request=req))
            continue
        items = parse_judge(resp.text)
        issues = (
            judge_issues(case, calc, note.text, items)
            if items is not None
            else [
                ValidationIssue(
                    param=None, source="judge", severity="error", problem="judge output unparseable"
                )
            ]
        )
        results.append(
            JudgeResult(
                case_id=note.case_id,
                render=render,
                locale=note.locale,
                judge_model=pcfg.model,
                parsed=items is not None,
                issues=issues,
                raw=resp.text,
            )
        )
    write_jsonl(run_dir(cfg) / "notes" / f"{render}.judge.jsonl", results)
    _finish_pending(pending_path(cfg, "judge", render), pending)
    return {
        "judged": len(results),
        "pending": len(pending),
        "notes_with_judge_errors": sum(
            any(i.severity == "error" for i in r.issues) for r in results
        ),
    }


def import_session_responses(
    cfg: RunConfig, pending: Path, responses: Path
) -> tuple[int, list[str]]:
    return import_responses(pending, responses, DiskCache(cfg.cache_dir))


def export_review(cfg: RunConfig, render: str) -> Path:
    cases = {c.case_id: c for c in read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)}
    notes = read_jsonl(notes_path(cfg, render), RenderedNote)
    judge_file = run_dir(cfg) / "notes" / f"{render}.judge.jsonl"
    judged = (
        {(r.case_id, r.locale): r for r in read_jsonl(judge_file, JudgeResult)}
        if judge_file.exists()
        else {}
    )
    md = export_review_sample(
        notes,
        cases,
        calculators(cfg),
        judged,
        cfg.validation.review_fraction,
        stable_seed(cfg.seed, "review", render),
    )
    out = run_dir(cfg) / "review" / f"{render}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return out
