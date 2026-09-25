"""Command-line entry point: `uv run calc-bounds <command> CONFIG`."""

from pathlib import Path

import typer

from calc_bounds import pipeline
from calc_bounds.config import load_config

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
def render(config: Path) -> None:
    """Render notes for each locale and run the validation pass."""
    _todo("M3")


@app.command()
def export_review(config: Path) -> None:
    """Export a random review sample as Markdown."""
    _todo("M3")


@app.command()
def run(config: Path) -> None:
    """Run the configured policies over the cohort and write traces."""
    typer.echo(f"wrote {pipeline.run_policies(load_config(config))}")


@app.command(name="eval")
def evaluate(config: Path) -> None:
    """Compute metrics and plots from traces."""
    out = pipeline.evaluate(load_config(config))
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
def anchor(config: Path) -> None:
    """Run the MedCalc-Bench anchor."""
    _todo("M4")
