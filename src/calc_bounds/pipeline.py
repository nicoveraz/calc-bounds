"""Pipeline stages driven by a RunConfig. Outputs go to <output_dir>/<run_id>/."""

from pathlib import Path

import numpy as np

from calc_bounds.bounds import from_extractions, score_bounds
from calc_bounds.calculators import REGISTRY, Calculator, get_calculator
from calc_bounds.cohort import PatientCase, generate_cohort
from calc_bounds.cohort.generate import stable_seed
from calc_bounds.config import ProviderConfig, RunConfig
from calc_bounds.eval import metrics, plots
from calc_bounds.extraction import ExtractionResult, Extractor
from calc_bounds.extraction.calibration import Calibrator
from calc_bounds.extraction.llm import (
    EXTRACT_SYSTEM,
    extraction_prompt,
    extraction_schema,
    parse_extraction,
)
from calc_bounds.extraction.oracle import OracleExtractor, PrecomputedExtractor
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.llm import LLM, DiskCache, LLMRequest, PendingResponseError, cache_key
from calc_bounds.llm.clients import make_client
from calc_bounds.llm.session import PendingItem, import_responses, write_pending
from calc_bounds.policies import POLICIES, Policy, Trace
from calc_bounds.render import JudgeResult, RenderedNote, ValidationIssue
from calc_bounds.render.export import export_review_sample
from calc_bounds.render.prompts import (
    JUDGE_SYSTEM,
    PROMPT_VERSION,
    RENDER_SYSTEM,
    judge_prompt,
    render_prompt,
)
from calc_bounds.render.validate import judge_issues, parse_judge, rule_issues
from calc_bounds.simulator import SimulatedClinician, draw_unavailable
from calc_bounds.units import to_canonical


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
    x, r = cfg.extraction.extractor, cfg.extraction.render
    assert x is not None and r is not None
    results = read_jsonl(extractions_path(cfg, x, r), ExtractionResult)
    return PrecomputedExtractor(f"{x}__{r}", {res.case_id: res for res in results})


def load_calibrator(cfg: RunConfig) -> Calibrator:
    """The calibrator fitted on the dev split for the configured extraction (none for the
    oracle, whose confidence is 1)."""
    import json

    if cfg.extraction.kind == "oracle" or cfg.extraction.calibration == "none":
        return Calibrator(method="none")
    # Calibration belongs to the extraction, not to the clinician condition.
    path = (
        run_dir(cfg) / "calibration" / (f"{cfg.extraction.extractor}__{cfg.extraction.render}.json")
    )
    report = json.loads(path.read_text())
    return Calibrator.model_validate(report[cfg.extraction.calibration]["calibrator"])


def make_policies(cfg: RunConfig) -> dict[str, Policy]:
    from calc_bounds.cohort.generate import _priors
    from calc_bounds.policies.voi_echo import VoiEchoPolicy

    priors = {c: _priors(calc, cfg.cohort) for c, calc in calculators(cfg).items()}
    policies: dict[str, Policy] = dict(POLICIES) | {
        "s4_bounds_voi_echo": VoiEchoPolicy(priors, load_calibrator(cfg), cfg.echo_threshold)
    }
    if cfg.agent is not None:
        from calc_bounds.policies.llm_agent import LLMAgentPolicy

        pcfg = cfg.providers[cfg.agent.provider]
        policies["s2_llm_agent"] = LLMAgentPolicy(
            make_llm(cfg),
            cfg.agent.provider,
            pcfg.request_params(),
            pcfg.model,
            pcfg.kind,
            max_turns=cfg.agent.max_turns,
        )
    return policies


def extraction_label(cfg: RunConfig) -> str:
    base = (
        "oracle"
        if cfg.extraction.kind == "oracle"
        else f"{cfg.extraction.extractor}__{cfg.extraction.render}"
    )
    # Non-ideal clinicians get their own traces/eval (the ideal results keep their names).
    return base if cfg.simulator.name == "ideal" else f"{base}__{cfg.simulator.name}"


def make_clinician(cfg: RunConfig, case: PatientCase, calc: Calculator) -> SimulatedClinician:
    unavailable = draw_unavailable(
        [p.id for p in calc.parameters],
        cfg.simulator.unavailable_rate,
        stable_seed(cfg.seed, case.case_id, "simulator"),
    )
    return SimulatedClinician(
        case.truth,
        unavailable,
        specs={p.id: p for p in calc.parameters},
        noise=cfg.simulator.noise,
        seed=stable_seed(cfg.seed, case.case_id, "clinician-noise"),
    )


def traces_path(cfg: RunConfig) -> Path:
    return run_dir(cfg) / "traces" / f"{extraction_label(cfg)}.jsonl"


def _drop_failed_notes(cfg: RunConfig, cases: list[PatientCase]) -> list[PatientCase]:
    """Primary analysis excludes cases whose note still fails validation after all render
    attempts (never silently: the render stats report them)."""
    render = cfg.extraction.render
    if render is None or not notes_path(cfg, render).exists():
        return cases
    failed = {
        n.case_id for n in read_jsonl(notes_path(cfg, render), RenderedNote) if n.validation_failed
    }
    return [c for c in cases if c.case_id not in failed]


def run_policies(cfg: RunConfig) -> Path:
    """Run the configured policies. Code policies run per case; the S2 agent runs in parallel
    across cases (its turns within a case are sequential LLM calls)."""
    from concurrent.futures import ThreadPoolExecutor

    out = run_dir(cfg)
    cases = read_jsonl(out / "cohort.jsonl", PatientCase)
    calcs = calculators(cfg)
    extractor = _extractor(cfg, cases)
    if isinstance(extractor, PrecomputedExtractor):  # e.g. a pilot subset
        cases = [c for c in cases if c.case_id in extractor.results]
    cases = _drop_failed_notes(cfg, cases)
    policies = make_policies(cfg)
    missing = [p for p in cfg.policies if p not in policies]
    if missing:
        raise NotImplementedError(f"policies not implemented yet: {missing}")
    notes: dict[str, str] = {}
    if "s2_llm_agent" in cfg.policies:
        assert cfg.agent is not None, "s2_llm_agent needs an `agent` config"
        # The agent reads the same notes the extraction came from (e.g. the messy set).
        agent_render = cfg.extraction.render or cfg.agent.render
        notes = {n.case_id: n.text for n in read_jsonl(notes_path(cfg, agent_render), RenderedNote)}
        cases = [c for c in cases if c.case_id in notes]

    def clinician_for(case: PatientCase) -> SimulatedClinician:
        return make_clinician(cfg, case, calcs[case.calculator])

    def run_one(pid: str, case: PatientCase) -> Trace:
        note = notes.get(case.case_id, "")  # code policies read the (precomputed) extraction
        return policies[pid].run(case, note, calcs[case.calculator], extractor, clinician_for(case))

    traces: list[Trace] = []
    for pid in cfg.policies:
        if pid == "s2_llm_agent":
            assert cfg.agent is not None
            with ThreadPoolExecutor(max_workers=cfg.agent.max_workers) as pool:
                traces += list(pool.map(lambda c: run_one("s2_llm_agent", c), cases))
        else:
            traces += [run_one(pid, c) for c in cases]
    write_jsonl(traces_path(cfg), traces)
    return traces_path(cfg)


def evaluate(cfg: RunConfig) -> Path:
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    traces = read_jsonl(traces_path(cfg), Trace)
    out = run_dir(cfg) / "eval" / extraction_label(cfg)
    out.mkdir(parents=True, exist_ok=True)
    table = metrics.case_table(cases, traces)
    table.to_csv(out / "cases.csv", index=False)
    overall = metrics.summary(table, ["policy"])
    per_calc = metrics.summary(table, ["policy", "calculator"])
    by_coverage = metrics.summary(table, ["policy", "determined_from_note"])
    overall.to_csv(out / "summary.csv", index=False)
    per_calc.to_csv(out / "summary_by_calculator.csv", index=False)
    by_coverage.to_csv(out / "summary_by_coverage.csv", index=False)
    ext = metrics.extraction_table(cases, traces)
    ext.to_csv(out / "extraction_claims.csv", index=False)
    from calc_bounds.eval.attribution import attribution_table
    from calc_bounds.eval.stats import comparisons

    comparisons(table).to_csv(out / "comparisons.csv", index=False)
    attr = attribution_table(cases, traces)
    attr.to_csv(out / "errors.csv", index=False)
    attr.groupby(["policy", "cause"]).size().rename("n").reset_index().to_csv(
        out / "error_attribution.csv", index=False
    )
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
    # Cases with identical fact sheets produce identical requests: answer each only once.
    pending = list({it.key: it for it in pending}.values())
    if pending:
        write_pending(path, pending)
    elif path.exists():
        path.unlink()


def _render_request(
    pcfg: ProviderConfig,
    case: PatientCase,
    calc: Calculator,
    locale: str,
    attempt: int,
    style: str = "standard",
) -> LLMRequest:
    req = _request(pcfg, RENDER_SYSTEM, render_prompt(case, calc, locale, style))
    if attempt > 1:  # attempt 1 carries no marker, so its cache key is the plain prompt's
        req.params["attempt"] = attempt
    return req


def _judge_provider(cfg: RunConfig, render: str) -> str | None:
    return cfg.renders[render].judge_provider or cfg.validation.judge_provider


def _judge_request(cfg: RunConfig, render: str, calc: Calculator, text: str) -> LLMRequest | None:
    judge = _judge_provider(cfg, render)
    if judge is None:
        return None
    return _request(cfg.providers[judge], JUDGE_SYSTEM, judge_prompt(text, calc))


def _judge_verdict(
    cfg: RunConfig, render: str, llm: LLM, case: PatientCase, calc: Calculator, text: str
) -> list[ValidationIssue] | None:
    """Judge issues from the cache only (never calls a model); None if not judged yet."""
    req = _judge_request(cfg, render, calc, text)
    if req is None:
        return []
    cached = llm.cache.get(cache_key(req))
    if cached is None:
        return None
    items = parse_judge(cached.text)
    if items is None:
        return [
            ValidationIssue(
                param=None, source="judge", severity="error", problem="judge output unparseable"
            )
        ]
    return judge_issues(case, calc, text, items)


def _failed(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)


def render_notes(cfg: RunConfig, render: str) -> dict[str, int]:
    """Render every selected case x locale, reading from the cache where possible.

    Attempts 1..max_attempts are walked in order: the first attempt that passes validation
    (rule checks + cached judge verdict) is kept. The walk stops at an attempt that is not yet
    rendered (exported as pending for session providers) or not yet judged (run `judge`, then
    `render` again). If every attempt fails, the last one is kept with validation_failed=True.
    """
    spec = cfg.renders[render]
    pcfg = cfg.providers[spec.provider]
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    calcs = calculators(cfg)
    llm = make_llm(cfg)

    def render_one(
        case: PatientCase, locale: str
    ) -> tuple[RenderedNote | None, PendingItem | None]:
        calc = calcs[case.calculator]
        for attempt in range(1, spec.max_attempts + 1):
            req = _render_request(pcfg, case, calc, locale, attempt, spec.style)
            try:
                resp = llm.complete(req, provider=spec.provider, stage=f"render:{render}")
            except PendingResponseError as e:
                return None, PendingItem(key=e.key, request=req)
            if resp.stop_reason == "error":
                return None, None  # provider failure (not cached): retried on the next run
            rules = rule_issues(case, calc, resp.text, locale)
            verdict = (
                None if _failed(rules) else _judge_verdict(cfg, render, llm, case, calc, resp.text)
            )
            failed = _failed(rules) or (verdict is not None and _failed(verdict))
            # Not judged yet counts as passing for now: `judge` runs next, and a later
            # `render` re-checks and retries if the verdict fails.
            if not failed or attempt == spec.max_attempts:
                return (
                    RenderedNote(
                        case_id=case.case_id,
                        render=render,
                        locale=locale,
                        text=resp.text,
                        provider=pcfg.kind,
                        model=pcfg.model,
                        cache_key=cache_key(req),
                        prompt_version=PROMPT_VERSION,
                        attempt=attempt,
                        validation_failed=failed,
                        issues=rules,
                    ),
                    None,
                )
        return None, None

    from concurrent.futures import ThreadPoolExecutor

    tasks = [
        (c, loc)
        for c in select_cases(cfg, cases, spec.subset_per_calculator)
        for loc in spec.locales
    ]
    with ThreadPoolExecutor(max_workers=spec.max_workers) as pool:
        outcomes = list(pool.map(lambda t: render_one(*t), tasks))
    notes = [n for n, _ in outcomes if n is not None]
    pending = [p for _, p in outcomes if p is not None]
    write_jsonl(notes_path(cfg, render), notes)
    _finish_pending(pending_path(cfg, "render", render), pending)
    return {
        "notes": len(notes),
        "pending": len({it.key for it in pending}),
        "retried_notes": sum(n.attempt > 1 for n in notes),
        "notes_failing_after_max_attempts": sum(n.validation_failed for n in notes),
        "notes_with_rule_errors": sum(_failed(n.issues) for n in notes),
    }


def judge_notes(cfg: RunConfig, render: str) -> dict[str, int]:
    judge = _judge_provider(cfg, render)
    if judge is None:
        raise ValueError("no judge provider configured")
    pcfg = cfg.providers[judge]
    if pcfg.model == cfg.providers[cfg.renders[render].provider].model:
        raise ValueError("judge model must differ from the renderer model")
    cases = {c.case_id: c for c in read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)}
    calcs = calculators(cfg)
    llm = make_llm(cfg)

    def judge_one(note: RenderedNote) -> tuple[JudgeResult | None, PendingItem | None]:
        case = cases[note.case_id]
        calc = calcs[case.calculator]
        req = _judge_request(cfg, render, calc, note.text)
        assert req is not None
        try:
            resp = llm.complete(req, provider=judge, stage=f"judge:{render}")
        except PendingResponseError as e:
            return None, PendingItem(key=e.key, request=req)
        if resp.stop_reason == "error":
            return None, None
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
        return (
            JudgeResult(
                case_id=note.case_id,
                render=render,
                locale=note.locale,
                judge_model=pcfg.model,
                parsed=items is not None,
                issues=issues,
                raw=resp.text,
            ),
            None,
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=cfg.renders[render].max_workers) as pool:
        outcomes = list(pool.map(judge_one, read_jsonl(notes_path(cfg, render), RenderedNote)))
    results = [r for r, _ in outcomes if r is not None]
    pending = [p for _, p in outcomes if p is not None]
    write_jsonl(run_dir(cfg) / "notes" / f"{render}.judge.jsonl", results)
    _finish_pending(pending_path(cfg, "judge", render), pending)
    return {
        "judged": len(results),
        "pending": len({it.key for it in pending}),
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


# --- M4: LLM extraction and calibration ---------------------------------------------------------


def extractions_path(cfg: RunConfig, extractor: str, render: str) -> Path:
    return run_dir(cfg) / "extractions" / f"{extractor}__{render}.jsonl"


def _extraction_request(cfg: RunConfig, extractor: str, calc: Calculator, note: str) -> LLMRequest:
    pcfg = cfg.providers[cfg.extractors[extractor].provider]
    req = _request(pcfg, EXTRACT_SYSTEM, extraction_prompt(note, calc))
    req.params["format"] = extraction_schema(calc)
    if pcfg.kind == "ollama":
        req.params["logprobs"] = True
        req.params["top_logprobs"] = 3
    return req


def extract_notes(
    cfg: RunConfig, extractor: str, render: str, per_calc: int | None = None
) -> dict[str, int]:
    """Run an LLM extractor over a render set's notes (cache-first, concurrent). Notes shared
    by several cases are extracted once. `per_calc` restricts to the nested subset used for
    pilots."""
    from concurrent.futures import ThreadPoolExecutor

    xcfg = cfg.extractors[extractor]
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    keep = {c.case_id for c in select_cases(cfg, cases, per_calc)}
    by_id = {c.case_id: c for c in cases}
    calcs = calculators(cfg)
    notes = [n for n in read_jsonl(notes_path(cfg, render), RenderedNote) if n.case_id in keep]
    llm = make_llm(cfg)

    def work(note: RenderedNote) -> ExtractionResult:
        calc = calcs[by_id[note.case_id].calculator]
        req = _extraction_request(cfg, extractor, calc, note.text)
        resp = llm.complete(req, provider=xcfg.provider, stage=f"extract:{extractor}:{render}")
        result = parse_extraction(note.case_id, note.text, calc, resp.text, resp.logprobs)
        return result.model_copy(
            update={"usage": resp.usage, "extractor": extractor, "render": render}
        )

    # Unique texts first (identical notes hit the cache), preserving case order in the output.
    with ThreadPoolExecutor(max_workers=xcfg.max_workers) as pool:
        results = list(pool.map(work, notes))
    out = extractions_path(cfg, extractor, render)
    write_jsonl(out, results)
    return {
        "notes": len(results),
        "rejected_claims": sum(len(r.rejected) for r in results),
        "cached": sum(bool(r.usage and r.usage.cached) for r in results),
    }


def dev_cases(cfg: RunConfig, cases: list[PatientCase]) -> set[str]:
    """Seeded per-calculator dev split (for calibration fitting only)."""
    dev: set[str] = set()
    for calc_id in cfg.calculators:
        ids = sorted(c.case_id for c in cases if c.calculator == calc_id)
        rng = np.random.default_rng(stable_seed(cfg.seed, "dev-split", calc_id))
        k = round(cfg.extraction.dev_fraction * len(ids))
        dev |= {ids[i] for i in rng.permutation(len(ids))[:k]}
    return dev


def calibrate(cfg: RunConfig, extractor: str, render: str) -> Path:
    """Fit calibrators on the dev split; report Brier/ECE before and after on the test split."""
    import json

    from calc_bounds.eval.metrics import claim_correct
    from calc_bounds.extraction import calibration as cal

    cases = {c.case_id: c for c in read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)}
    results = read_jsonl(extractions_path(cfg, extractor, render), ExtractionResult)
    dev = dev_cases(cfg, list(cases.values()))
    rows = []
    for r in results:
        for pid, e in r.values.items():
            if e.kind == "unknown":
                continue
            rows.append(
                {
                    "case_id": r.case_id,
                    "dev": r.case_id in dev,
                    "source": e.confidence_source,  # type: ignore[union-attr]
                    "confidence": e.confidence,  # type: ignore[union-attr]
                    "correct": claim_correct(cases[r.case_id], pid, e),
                }
            )
    report: dict[str, object] = {"extractor": extractor, "render": render, "n_claims": len(rows)}
    d = [x for x in rows if x["dev"]]
    t = [x for x in rows if not x["dev"]]
    tc = np.array([x["confidence"] for x in t])
    ty = np.array([x["correct"] for x in t])
    for method in ("none", "temperature", "isotonic"):
        c = cal.fit(method, [x["confidence"] for x in d], [x["correct"] for x in d])
        p = c.transform(tc)
        report[method] = {
            "calibrator": c.model_dump(),
            "test_brier": cal.brier(p, ty),
            "test_ece": cal.ece(p, ty),
            "test_reliability": cal.reliability(p, ty),
        }
    report["confidence_sources"] = sorted({str(x["source"]) for x in rows})
    report["test_accuracy_of_claims"] = float(ty.mean()) if len(ty) else None
    out = run_dir(cfg) / "calibration" / f"{extractor}__{render}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out


# --- M4: MedCalc-Bench anchor -------------------------------------------------------------------

ANCHOR_SPLITS = {
    "test": Path("data/raw/medcalc/test_data.csv"),
    "train": Path("data/raw/medcalc/train_data.csv"),
}


def _point_score(calc: Calculator, known: dict) -> float | None:
    b = score_bounds(calc, known)
    return b.lo if b.lo == b.hi else None


def anchor_run(
    cfg: RunConfig, extractor: str, split: str = "test", per_calc: int | None = None
) -> Path:
    """Extraction + code on MedCalc-Bench notes; see calc_bounds.anchor and docs/ANCHOR.md.
    `split=train` excludes notes that also appear in the test split; `per_calc` caps each
    calculator with a seeded sample."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    from calc_bounds.anchor import AnchorCase, load_anchor
    from calc_bounds.bounds import Exact

    calcs = {c: get_calculator(c, cfg.calculator_options.get(c)) for c in REGISTRY}
    xcfg = cfg.extractors[extractor]
    llm = make_llm(cfg)
    cases = load_anchor(ANCHOR_SPLITS[split])
    if split != "test":
        test_notes = {a.note for a in load_anchor(ANCHOR_SPLITS["test"])}
        cases = [a for a in cases if a.note not in test_notes]
    if per_calc is not None:
        kept = []
        for calc_id in sorted({a.calculator for a in cases}):
            pool = [a for a in cases if a.calculator == calc_id]
            rng = np.random.default_rng(stable_seed(cfg.seed, "anchor", split, calc_id))
            kept += [pool[i] for i in sorted(rng.permutation(len(pool))[:per_calc])]
        cases = kept

    def work(a: AnchorCase) -> dict:
        calc = calcs[a.calculator]
        req = _extraction_request(cfg, extractor, calc, a.note)
        resp = llm.complete(req, provider=xcfg.provider, stage=f"anchor:{extractor}")
        ext = parse_extraction(str(a.row_number), a.note, calc, resp.text, resp.logprobs)
        gt_cat = calc.category(a.ground_truth)
        # (1) our code on their annotated entities, MedCalc convention (missing -> normal).
        theirs = from_extractions(calc, {}, binary=True) | {
            p: Exact(value=v) for p, v in a.entities.items()
        }
        ours_on_theirs = _point_score(calc, theirs)
        # (2) our extraction, MedCalc convention.
        conv = _point_score(calc, from_extractions(calc, ext.values, binary=True))
        # (3) our extraction, tri-state bounds.
        tri = score_bounds(calc, from_extractions(calc, ext.values))
        # (4) parameter agreement with annotated entities.
        agree = total = 0
        for pid, v in a.entities.items():
            e = ext.values[pid]
            total += 1
            if isinstance(v, bool):
                agree += (e.kind == "present") == v
            elif e.kind == "present":
                got = e.value  # type: ignore[union-attr]
                unit = e.unit  # type: ignore[union-attr]
                got = to_canonical(pid, float(got), unit) if unit else float(got)
                agree += abs(got - float(v)) <= 0.02 * max(abs(float(v)), 1e-9)
        return {
            "row_number": a.row_number,
            "calculator": a.calculator,
            "note_type": a.note_type,
            "ground_truth": a.ground_truth,
            "gt_category": gt_cat,
            "ours_on_their_entities": ours_on_theirs,
            "impl_agrees": ours_on_theirs is not None and a.lower <= ours_on_theirs <= a.upper,
            "medcalc_convention_score": conv,
            "medcalc_convention_correct": conv is not None and a.lower <= conv <= a.upper,
            "medcalc_convention_category_correct": conv is not None
            and calc.category(conv) == gt_cat,
            "tristate_determined": len(tri.categories) == 1,
            "tristate_category_correct": tri.categories == {gt_cat},
            "tristate_gt_possible": gt_cat in tri.categories,
            "entity_agreement": agree / total if total else None,
            "rejected_claims": len(ext.rejected),
        }

    with ThreadPoolExecutor(max_workers=xcfg.max_workers) as pool:
        rows = list(pool.map(work, cases))
    name = extractor if split == "test" else f"{extractor}__{split}"
    out = run_dir(cfg) / "anchor" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    cols = [
        "impl_agrees",
        "medcalc_convention_correct",
        "medcalc_convention_category_correct",
        "tristate_determined",
        "tristate_category_correct",
        "tristate_gt_possible",
        "entity_agreement",
    ]
    df.groupby("calculator")[cols].mean().round(3).to_csv(out.with_suffix(".summary.csv"))
    return out


def renderer_bias_report(
    cfg: RunConfig, extractors: tuple[str, str], renders: tuple[str, str]
) -> Path:
    """2x2 renderer family x extractor family on cases whose notes pass validation in both
    render sets (option 1 agreed with the user: exploratory, small n)."""
    import json

    from calc_bounds.eval.bias import claim_frame, renderer_bias

    cases = {c.case_id: c for c in read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)}
    valid: set[str] | None = None
    for r in renders:
        ok = {
            n.case_id
            for n in read_jsonl(notes_path(cfg, r), RenderedNote)
            if not n.validation_failed
        }
        valid = ok if valid is None else valid & ok
    assert valid is not None
    results = {
        (x, r): read_jsonl(extractions_path(cfg, x, r), ExtractionResult)
        for x in extractors
        for r in renders
    }
    report = renderer_bias(claim_frame(cases, results, valid), extractors, renders)
    report["by_calculator_n"] = {
        c: sum(cases[i].calculator == c for i in valid) for c in cfg.calculators
    }
    out = run_dir(cfg) / "eval" / "renderer_bias.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out


# --- M6: cross-source report --------------------------------------------------------------------

ECHO_THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
MISSINGNESS_LEVELS = [0.1, 0.3, 0.5]


def _cfg_for(cfg: RunConfig, label: str) -> RunConfig:
    """Config for a trace label: `oracle[__<clinician>]` or
    `<extractor>__<render>[__<clinician>]`."""
    parts = label.split("__")
    if parts[0] == "oracle":
        x = cfg.extraction.model_copy(update={"kind": "oracle", "extractor": None, "render": None})
        clinician = parts[1] if len(parts) > 1 else None
    else:
        x = cfg.extraction.model_copy(
            update={"kind": "llm", "extractor": parts[0], "render": parts[1]}
        )
        clinician = parts[2] if len(parts) > 2 else None
    update: dict[str, object] = {"extraction": x.model_dump()}
    if clinician:
        update["simulator"] = (
            cfg.clinicians[clinician].model_copy(update={"name": clinician}).model_dump()
        )
    return RunConfig.model_validate(cfg.model_dump() | update)


def report_sources(cfg: RunConfig) -> list[str]:
    """Every condition with traces: extraction source x note set x clinician."""
    order = {"oracle": 0}
    labels = sorted(p.stem for p in (run_dir(cfg) / "traces").glob("*.jsonl"))
    return sorted(labels, key=lambda lab: (order.get(lab.split("__")[0], 1), lab))


def report(cfg: RunConfig, missingness_n: int = 200) -> Path:
    """Aggregate English results across extraction sources (code-only; S2 traces come from
    the cache). Writes runs/<run>/report/."""
    import pandas as pd

    from calc_bounds.eval import metrics, plots
    from calc_bounds.eval.report import (
        natural_undetermined_share,
        reliability_figure,
        reweight,
    )

    out = run_dir(cfg) / "report"
    out.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(run_dir(cfg) / "cohort.jsonl", PatientCase)
    calcs = calculators(cfg)
    shares = {
        c: natural_undetermined_share(calc, cfg.cohort, cfg.seed) for c, calc in calcs.items()
    }
    pd.Series(shares, name="natural_undetermined_share").to_csv(out / "natural_share.csv")

    summaries, comps, attrs, natural, sweep, claims = [], [], [], [], [], {}
    for label in report_sources(cfg):
        c = _cfg_for(cfg, label)
        ev = evaluate(c)
        s = pd.read_csv(ev / "summary.csv")
        s.insert(0, "extraction", label)
        summaries.append(s)
        cp = pd.read_csv(ev / "comparisons.csv")
        cp.insert(0, "extraction", label)
        comps.append(cp)
        at = pd.read_csv(ev / "error_attribution.csv")
        at.insert(0, "extraction", label)
        attrs.append(at)
        table = metrics.case_table(cases, read_jsonl(traces_path(c), Trace))
        by_cov = (
            table.groupby(["policy", "calculator", "determined_from_note"])[
                ["correct", "n_questions"]
            ]
            .mean()
            .reset_index()
        )
        nat = reweight(by_cov, shares, "correct").merge(
            reweight(by_cov, shares, "n_questions"),
            on=["policy", "calculator", "natural_undetermined"],
        )
        nat.insert(0, "extraction", label)
        natural.append(nat)
        if not label.startswith("oracle") and label.count("__") == 1 and label.endswith("__sonnet"):
            claims[label.split("__")[0]] = pd.read_csv(ev / "extraction_claims.csv").query(
                "policy == 's3_bounds'"
            )
            for tau in ECHO_THRESHOLDS:
                ct = c.model_copy(
                    update={"echo_threshold": tau, "policies": ["s4_bounds_voi_echo"]}
                )
                ct = RunConfig.model_validate(ct.model_dump())
                tr = _run_code_policies(ct, cases)
                t = metrics.case_table(cases, tr)
                sweep.append(
                    {
                        "extraction": label,
                        "echo_threshold": tau,
                        "accuracy": t["correct"].mean(),
                        "mean_questions": t["n_questions"].mean(),
                        "echo_per_case": t["n_echo_questions"].mean(),
                        "echo_caught_errors": int(t["n_echo_caught_errors"].sum()),
                    }
                )
    pd.concat(summaries).to_csv(out / "summary_all.csv", index=False)
    pd.concat(comps).to_csv(out / "comparisons_all.csv", index=False)
    pd.concat(attrs).to_csv(out / "error_attribution_all.csv", index=False)
    pd.concat(natural).to_csv(out / "natural_share_reweighted.csv", index=False)
    if sweep:
        pd.DataFrame(sweep).to_csv(out / "s4_echo_threshold_sweep.csv", index=False)
    if claims:
        reliability_figure(claims, out / "reliability.png")

    # Missingness sensitivity (oracle extraction, code policies, fresh cohorts).
    rows = []
    for m in MISSINGNESS_LEVELS:
        cc = cfg.cohort.model_copy(update={"missingness": m, "n_per_calculator": missingness_n})
        mcfg = RunConfig.model_validate(
            cfg.model_dump()
            | {
                "cohort": cc.model_dump(),
                "policies": ["s1_ask_all", "s3_bounds", "s3_bin", "s4_bounds_voi_echo"],
                "extraction": {"kind": "oracle"},
            }
        )
        mcases = generate_cohort(list(calcs.values()), cc, cfg.seed)
        t = metrics.case_table(mcases, _run_code_policies(mcfg, mcases))
        mshares = {
            c: natural_undetermined_share(calc, cc, cfg.seed, n=1000) for c, calc in calcs.items()
        }
        by_cov = (
            t.groupby(["policy", "calculator", "determined_from_note"])[["correct", "n_questions"]]
            .mean()
            .reset_index()
        )
        nat = reweight(by_cov, mshares, "correct").merge(
            reweight(by_cov, mshares, "n_questions"),
            on=["policy", "calculator", "natural_undetermined"],
        )
        for pol, g in nat.groupby("policy"):
            rows.append(
                {
                    "missingness": m,
                    "policy": pol,
                    "accuracy_natural": g["correct"].mean(),
                    "questions_natural": g["n_questions"].mean(),
                    "mean_natural_undetermined": g["natural_undetermined"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out / "missingness_sensitivity.csv", index=False)

    s_all = pd.concat(summaries)
    for label in s_all["extraction"].unique():
        s = s_all[s_all["extraction"] == label]
        plots.accuracy_vs_questions(
            s.drop(columns="extraction"),
            metrics.summary(
                metrics.case_table(cases, read_jsonl(traces_path(_cfg_for(cfg, label)), Trace)),
                ["policy", "calculator"],
            ),
            out / f"accuracy_vs_questions__{label}.png",
        )
    return out


def _run_code_policies(cfg: RunConfig, cases: list[PatientCase]) -> list[Trace]:
    """Run code-only policies over the given cases (no LLM calls)."""
    calcs = calculators(cfg)
    extractor = _extractor(cfg, cases)
    if isinstance(extractor, PrecomputedExtractor):
        cases = [c for c in cases if c.case_id in extractor.results]
    policies = make_policies(cfg)
    traces = []
    for case in cases:
        calc = calcs[case.calculator]
        for pid in cfg.policies:
            if pid == "s2_llm_agent":
                continue
            traces.append(
                policies[pid].run(case, "", calc, extractor, make_clinician(cfg, case, calc))
            )
    return traces
