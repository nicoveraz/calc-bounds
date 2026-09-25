"""Pipeline stages driven by a RunConfig. Outputs go to <output_dir>/<run_id>/."""

from pathlib import Path

from calc_bounds.calculators import Calculator, get_calculator
from calc_bounds.cohort import PatientCase, generate_cohort
from calc_bounds.cohort.generate import stable_seed
from calc_bounds.config import RunConfig
from calc_bounds.eval import metrics, plots
from calc_bounds.extraction import Extractor
from calc_bounds.extraction.oracle import OracleExtractor
from calc_bounds.io import read_jsonl, write_jsonl
from calc_bounds.policies import POLICIES, Trace
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
