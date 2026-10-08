"""Command-line entry point: `uv run calc-bounds <command> CONFIG`."""

from pathlib import Path

import typer

from calc_bounds import pipeline
from calc_bounds.config import RunConfig, load_config

app = typer.Typer(no_args_is_help=True, help="calc-bounds experiment pipeline.")


@app.command()
def check_config(config: Path) -> None:
    """Validate a run config and print it."""
    typer.echo(load_config(config).model_dump_json(indent=2))


def _todo(milestone: str) -> None:
    typer.echo(f"Not implemented yet ({milestone}).", err=True)
    raise typer.Exit(code=1)


@app.command()
def cohort(config: Path) -> None:
    """Generate the synthetic cohort (JSONL)."""
    typer.echo(f"wrote {pipeline.make_cohort(load_config(config))}")


@app.command()
def render(
    config: Path, name: str = typer.Option(..., help="Render set (config `renders` key).")
) -> None:
    """Render notes (from cache where available) and run the rule-based validation pass.

    With a `session` provider, cache misses are exported to runs/<run>/pending/ for offline
    answering; import them with `import-responses`, then run `render` again."""
    stats = pipeline.render_notes(load_config(config), name)
    typer.echo(stats)
    if stats["pending"]:
        typer.echo(f"{stats['pending']} requests pending: answer them, then import-responses.")


@app.command()
def judge(config: Path, name: str = typer.Option(..., help="Render set to validate.")) -> None:
    """Semantic validation of rendered notes by a judge model (different from the renderer)."""
    typer.echo(pipeline.judge_notes(load_config(config), name))


@app.command()
def import_responses(config: Path, pending: Path, responses: Path) -> None:
    """Import offline (session) responses into the LLM cache."""
    n, problems = pipeline.import_session_responses(load_config(config), pending, responses)
    typer.echo(f"imported {n} responses")
    for p in problems:
        typer.echo(f"  problem: {p}", err=True)
    if problems:
        raise typer.Exit(code=1)


@app.command()
def export_review(config: Path, name: str = typer.Option(..., help="Render set.")) -> None:
    """Export a random review sample (validation.review_fraction) as Markdown."""
    typer.echo(f"wrote {pipeline.export_review(load_config(config), name)}")


def _load(
    config: Path, extractor: str | None, render: str | None, clinician: str | None = None
) -> RunConfig:
    """Load a config, optionally overriding the LLM extraction and the clinician condition."""
    cfg = load_config(config)
    if clinician:
        sim = cfg.clinicians[clinician].model_copy(update={"name": clinician})
        cfg = RunConfig.model_validate(cfg.model_dump() | {"simulator": sim.model_dump()})
    if extractor or render:
        x = cfg.extraction.model_copy(
            update={"kind": "llm", "extractor": extractor, "render": render}
        )
        cfg = RunConfig.model_validate(cfg.model_dump() | {"extraction": x.model_dump()})
    return cfg


EXTRACTOR_OPT = typer.Option(None, help="Use this LLM extractor's results (key in `extractors`).")
RENDER_OPT = typer.Option(None, help="Render set the extractor ran on.")
CLINICIAN_OPT = typer.Option(None, help="Named clinician condition (key in `clinicians`).")


@app.command()
def run(
    config: Path,
    extractor: str | None = EXTRACTOR_OPT,
    render: str | None = RENDER_OPT,
    clinician: str | None = CLINICIAN_OPT,
) -> None:
    """Run the configured policies over the cohort and write traces."""
    typer.echo(f"wrote {pipeline.run_policies(_load(config, extractor, render, clinician))}")


@app.command()
def extract(
    config: Path,
    extractor: str = typer.Option(..., help="Key in `extractors`."),
    render: str = typer.Option(..., help="Render set whose notes to extract."),
    per_calc: int | None = typer.Option(None, help="Pilot: nested subset per calculator."),
) -> None:
    """Run an LLM extractor over rendered notes (cache-first)."""
    typer.echo(pipeline.extract_notes(load_config(config), extractor, render, per_calc))


@app.command()
def calibrate(
    config: Path,
    extractor: str = typer.Option(...),
    render: str = typer.Option(...),
) -> None:
    """Fit calibration on the dev split and report Brier/ECE on the test split."""
    typer.echo(f"wrote {pipeline.calibrate(load_config(config), extractor, render)}")


@app.command(name="eval")
def evaluate(
    config: Path,
    extractor: str | None = EXTRACTOR_OPT,
    render: str | None = RENDER_OPT,
    clinician: str | None = CLINICIAN_OPT,
) -> None:
    """Compute metrics and plots from traces."""
    out = pipeline.evaluate(_load(config, extractor, render, clinician))
    typer.echo((out / "summary.csv").read_text())
    typer.echo(f"outputs in {out}")


@app.command()
def all(config: Path) -> None:
    """cohort -> run -> eval."""
    cfg = load_config(config)
    pipeline.make_cohort(cfg)
    pipeline.run_policies(cfg)
    out = pipeline.evaluate(cfg)
    typer.echo((out / "summary.csv").read_text())
    typer.echo(f"outputs in {out}")


@app.command()
def report(config: Path) -> None:
    """Cross-source English report: summaries, paired stats, attribution, S4 sweep,
    missingness sensitivity, figures (code-only; run after `run` for each source)."""
    typer.echo(f"wrote {pipeline.report(load_config(config))}")


PAPER_OUT_OPT = typer.Option(Path("paper"), help="Output directory.")


@app.command()
def paper(config: Path, out: Path = PAPER_OUT_OPT) -> None:
    """Regenerate Paper 1 tables and figures from run outputs (no model calls)."""
    from calc_bounds.eval.paper import build

    cfg = load_config(config)
    typer.echo(build(pipeline.run_dir(cfg), out, pipeline.calculators(cfg)))


@app.command()
def bias(
    config: Path,
    extractors: str = typer.Option("haiku,qwen_local", help="Two extractors, comma-separated."),
    renders: str = typer.Option("sonnet,local", help="Their families' render sets, same order."),
) -> None:
    """Renderer-family x extractor-family 2x2 on validated notes."""
    x = tuple(extractors.split(","))
    r = tuple(renders.split(","))
    out = pipeline.renderer_bias_report(load_config(config), x, r)  # type: ignore[arg-type]
    typer.echo(out.read_text())


@app.command()
def anchor(
    config: Path,
    extractor: str = typer.Option(..., help="Key in `extractors`."),
    split: str = typer.Option("test", help="MedCalc-Bench split: test or train."),
    per_calc: int | None = typer.Option(None, help="Seeded cap per calculator."),
) -> None:
    """Run the MedCalc-Bench anchor (extraction + code) with one extractor."""
    out = pipeline.anchor_run(load_config(config), extractor, split, per_calc)
    typer.echo(out.with_suffix(".summary.csv").read_text())


# --- MIMIC-IV validation (credentialed data outside the repo; local models only) ---------------

mimic_app = typer.Typer(
    no_args_is_help=True,
    help="MIMIC-IV validation. Data, row-level outputs and the LLM cache stay outside the repo; "
    "only local models are allowed. See docs/MIMIC_VALIDATION.md.",
)
app.add_typer(mimic_app, name="mimic")

MIMIC_EXTRACTOR_OPT = typer.Option("oracle", help="Key in `extractors`, or 'oracle' (no model).")
MIMIC_OUT_OPT = typer.Option(Path("results/mimic"), help="Where aggregate tables go.")


@mimic_app.command("check-config")
def mimic_check_config(config: Path) -> None:
    """Validate a MIMIC config (paths outside the repo, local providers only)."""
    from calc_bounds.mimic.config import load_mimic_config
    from calc_bounds.mimic.criteria import load_criteria

    cfg = load_mimic_config(config)
    load_criteria(cfg.criteria)
    typer.echo(cfg.model_dump_json(indent=2))


@mimic_app.command("cohort")
def mimic_cohort(config: Path) -> None:
    """Build cases: cohorts, structured truth and note sections (row-level, outside the repo)."""
    from calc_bounds.mimic import runner
    from calc_bounds.mimic.config import load_mimic_config

    out = runner.build_cohort(load_mimic_config(config))
    typer.echo((out / "cohort_flow.csv").read_text())
    typer.echo(f"wrote {out}")


@mimic_app.command("annotation-template")
def mimic_annotation_template(
    config: Path,
    n: int | None = typer.Option(None, help="Seeded sample of cases per calculator."),
) -> None:
    """Export the judgement-item template (ids only) for physician annotation."""
    from calc_bounds.mimic import runner
    from calc_bounds.mimic.config import load_mimic_config

    typer.echo(f"wrote {runner.export_annotations(load_mimic_config(config), n)}")


@mimic_app.command("import-annotations")
def mimic_import_annotations(config: Path, filled: Path) -> None:
    """Validate a filled annotation template and store it with the run."""
    from calc_bounds.mimic import runner
    from calc_bounds.mimic.config import load_mimic_config

    typer.echo(f"wrote {runner.import_annotations(load_mimic_config(config), filled)}")


@mimic_app.command("run")
def mimic_run(config: Path, extractor: str = MIMIC_EXTRACTOR_OPT) -> None:
    """Extraction -> S1 / S3 / S4 / S3-bin with the EHR as the clinician (row-level outputs)."""
    from calc_bounds.mimic import runner
    from calc_bounds.mimic.config import load_mimic_config

    typer.echo(f"wrote {runner.run(load_mimic_config(config), extractor)}")


@mimic_app.command("diagnose")
def mimic_diagnose(config: Path) -> None:
    """Aggregate-only diagnostics: troponin items in the window, GCS coverage."""
    from calc_bounds.mimic.config import load_mimic_config
    from calc_bounds.mimic.diagnose import gcs_report, troponin_report

    cfg = load_mimic_config(config)
    typer.echo(troponin_report(cfg).to_csv(index=False))
    typer.echo(gcs_report(cfg).to_csv(index=False))


@mimic_app.command("aggregate")
def mimic_aggregate(
    config: Path,
    extractor: str = MIMIC_EXTRACTOR_OPT,
    out: Path = MIMIC_OUT_OPT,
) -> None:
    """Aggregate tables only (counts, rates, Wilson CIs, means); safe to commit."""
    from calc_bounds.mimic import runner
    from calc_bounds.mimic.config import load_mimic_config

    typer.echo(f"wrote {runner.aggregate(load_mimic_config(config), extractor, out)}")
