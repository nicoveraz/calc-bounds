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


def _load(config: Path, extractor: str | None, render: str | None) -> RunConfig:
    """Load a config, optionally overriding which LLM extraction the policies use."""
    cfg = load_config(config)
    if extractor or render:
        x = cfg.extraction.model_copy(
            update={"kind": "llm", "extractor": extractor, "render": render}
        )
        cfg = RunConfig.model_validate(cfg.model_dump() | {"extraction": x.model_dump()})
    return cfg


EXTRACTOR_OPT = typer.Option(None, help="Use this LLM extractor's results (key in `extractors`).")
RENDER_OPT = typer.Option(None, help="Render set the extractor ran on.")


@app.command()
def run(
    config: Path, extractor: str | None = EXTRACTOR_OPT, render: str | None = RENDER_OPT
) -> None:
    """Run the configured policies over the cohort and write traces."""
    typer.echo(f"wrote {pipeline.run_policies(_load(config, extractor, render))}")


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
    config: Path, extractor: str | None = EXTRACTOR_OPT, render: str | None = RENDER_OPT
) -> None:
    """Compute metrics and plots from traces."""
    out = pipeline.evaluate(_load(config, extractor, render))
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
def anchor(config: Path, extractor: str = typer.Option(..., help="Key in `extractors`.")) -> None:
    """Run the MedCalc-Bench anchor (extraction + code) with one extractor."""
    out = pipeline.anchor_run(load_config(config), extractor)
    typer.echo(out.with_suffix(".summary.csv").read_text())
