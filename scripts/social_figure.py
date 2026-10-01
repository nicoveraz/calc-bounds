"""LinkedIn figure (1200 x 1500 px, 4:5 portrait), in English and Spanish: a frontier agent
working alone (Claude Opus 5.5) versus a local model plus code (Qwen3.5-9B on a laptop + the
bounds policy), on safety and cost.

All numbers are read from the run outputs in runs/main: the report summary, the per-case traces
(agent token usage and time) and the usage ledger (local reading time). Nothing is hand-entered.
Output: paper/social/.

    uv run python scripts/social_figure.py
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calc_bounds.eval.plots import R_RC

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "main"
OUT = ROOT / "paper" / "social"
FRONTIER = "#6B6B6B"
LOCAL = "#2297E6"  # R 4 palette blue


def _jsonl(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def measures() -> dict[str, tuple[float, float]]:
    """(frontier agent, local model + bounds) for each measure; clean notes, 1,200 cases."""
    s = pd.read_csv(RUN / "report" / "summary_all.csv").set_index(["extraction", "policy"])
    ideal, noisy = "qwen_local__sonnet", "qwen_local__sonnet__noisy"

    def pair(label: str, col: str, scale: float = 100.0) -> tuple[float, float]:
        return (
            scale * s.loc[(label, "s2_llm_agent"), col],
            scale * s.loc[(label, "s3_bounds"), col],
        )

    agent_usage = [
        r["usage"]
        for r in _jsonl(RUN / "traces" / f"{ideal}.jsonl")
        if r["policy"] == "s2_llm_agent" and r["usage"]
    ]
    tokens = statistics.mean(u["input_tokens"] + u["output_tokens"] for u in agent_usage)
    agent_s = statistics.median(u["latency_s"] for u in agent_usage if u["latency_s"] > 0)
    local_s = statistics.median(
        r["latency_s"]
        for r in _jsonl(RUN / "usage.jsonl")
        if r["stage"] == "extract:qwen_local:sonnet" and not r["cached"] and r["latency_s"] > 0
    )
    return {
        "accuracy": pair(ideal, "accuracy"),
        "early": pair(noisy, "premature_commitment_rate"),
        "irrelevant": pair(ideal, "irrelevant_question_rate"),
        "under": pair(noisy, "under_triage_rate"),
        "tokens": (tokens, 0.0),
        "questions": pair(ideal, "mean_questions", 1.0),
        "seconds": (agent_s, local_s),
    }


TEXT = {
    "en": {
        "title": "Frontier AI alone, or\nsmall local AI + code?",
        "subtitle": "Same accuracy. The local pipeline is safer, and it\n"
        "needs no paid model and no data leaving the computer.",
        "cols": ("Frontier agent\nClaude Opus 5.5", "Local model + code\nQwen3.5-9B on a laptop"),
        "safety": "SAFETY",
        "cost": "COST",
        "rows": {
            "accuracy": "Correct risk group",
            "early": "Answered before the\nanswer was settled*",
            "irrelevant": "Questions that could not\nchange the decision",
            "under": "Placed in a lower-risk\ngroup than the true one*",
            "tokens": "Frontier-model tokens\nper patient",
            "data": "Patient note leaves\nthe computer",
            "questions": "Questions to the\nclinician per patient",
            "seconds": "Seconds per patient\n(median)",
        },
        "yes": "Yes",
        "no": "No",
        "sep": ",",
        "dec": ".",
        "pct": "%",
        "foot": "Simulation: 1,200 synthetic emergency cases. *With a clinician who sometimes does\n"
        "not know, misremembers or answers vaguely; other rows with a clinician who answers\n"
        "correctly. Local reading time measured on an Apple M1 Pro laptop, 16 GB.",
        "cite": "Vera Zúñiga N. arXiv:2609.34112 · github.com/nicoveraz/calc-bounds",
    },
    "es": {
        "title": "¿IA de frontera sola, o\nIA local pequeña + código?",
        "subtitle": "Misma precisión. La alternativa local es más segura,\n"
        "sin modelo pagado y sin que los datos salgan del equipo.",
        "cols": (
            "Agente de frontera\nClaude Opus 5.5",
            "Modelo local + código\nQwen3.5-9B en un portátil",
        ),
        "safety": "SEGURIDAD",
        "cost": "COSTO",
        "rows": {
            "accuracy": "Grupo de riesgo correcto",
            "early": "Respondió antes de que la\nrespuesta estuviera definida*",
            "irrelevant": "Preguntas que no podían\ncambiar la decisión",
            "under": "Clasificado en un grupo de\nmenor riesgo que el real*",
            "tokens": "Tokens del modelo de\nfrontera por paciente",
            "data": "La ficha sale\ndel equipo",
            "questions": "Preguntas al clínico\npor paciente",
            "seconds": "Segundos por paciente\n(mediana)",
        },
        "yes": "Sí",
        "no": "No",
        "sep": ".",
        "dec": ",",
        "pct": " %",
        "foot": "Simulación: 1.200 casos sintéticos de urgencia. *Con un clínico que a veces no sabe,\n"
        "recuerda mal o responde vago; el resto, con un clínico que responde bien. Tiempo de\n"
        "lectura local medido en un portátil Apple M1 Pro de 16 GB.",
        "cite": "Vera Zúñiga N. arXiv:2609.34112 · github.com/nicoveraz/calc-bounds",
    },
}

# Which column is better for each row (shown bold and coloured); None = no clear winner.
BETTER = {
    "accuracy": None,
    "early": 1,
    "irrelevant": 1,
    "under": None,
    "tokens": 1,
    "data": 1,
    "questions": 0,
    "seconds": 0,
}
SECTIONS = [
    ("safety", ["accuracy", "early", "irrelevant", "under"]),
    ("cost", ["tokens", "data", "questions", "seconds"]),
]


def number(v: float, decimals: int, tx: dict) -> str:
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "\0").replace(".", tx["dec"]).replace("\0", tx["sep"])


def draw(lang: str, m: dict[str, tuple[float, float]]) -> Path:
    tx = TEXT[lang]
    cells = {
        k: [number(v, 1, tx) + tx["pct"] for v in m[k]]
        for k in ("accuracy", "early", "irrelevant", "under")
    }
    cells["tokens"] = ["≈" + number(round(m["tokens"][0], -2), 0, tx), "0"]
    cells["data"] = [tx["yes"], tx["no"]]
    cells["questions"] = [number(v, 2, tx) for v in m["questions"]]
    cells["seconds"] = [number(v, 0, tx) for v in m["seconds"]]
    with matplotlib.rc_context(R_RC):
        fig = plt.figure(figsize=(8, 10), dpi=150, facecolor="white")  # 1200 x 1500 px
        fig.text(
            0.06, 0.955, tx["title"], fontsize=29, fontweight="bold", va="top", linespacing=1.1
        )
        fig.text(0.06, 0.835, tx["subtitle"], fontsize=15, va="top", color="#333333")
        x_label, x_cols = 0.06, (0.58, 0.81)
        y = 0.74
        for i, head in enumerate(tx["cols"]):
            fig.text(
                x_cols[i],
                y,
                head,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color=FRONTIER if i == 0 else LOCAL,
                linespacing=1.15,
            )
        y -= 0.045
        for section, keys in SECTIONS:
            fig.add_artist(plt.Line2D([0.06, 0.94], [y, y], color="black", linewidth=1.0))
            fig.text(x_label, y - 0.01, tx[section], fontsize=12, fontweight="bold", va="top")
            y -= 0.048
            for key in keys:
                fig.text(x_label, y, tx["rows"][key], fontsize=13, va="center", linespacing=1.1)
                for i, cell in enumerate(cells[key]):
                    win = BETTER[key] == i
                    fig.text(
                        x_cols[i],
                        y,
                        cell,
                        fontsize=21 if win else 18,
                        fontweight="bold" if win else "normal",
                        ha="center",
                        va="center",
                        color=(FRONTIER if i == 0 else LOCAL) if win else "black",
                    )
                y -= 0.058
            y += 0.01
        fig.add_artist(plt.Line2D([0.06, 0.94], [y, y], color="black", linewidth=1.0))
        fig.text(
            0.06, y - 0.015, tx["foot"], fontsize=10.5, va="top", color="#333333", linespacing=1.35
        )
        fig.text(0.06, 0.04, tx["cite"], fontsize=11.5, va="top", fontweight="bold")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"linkedin_{lang}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


if __name__ == "__main__":
    values = measures()
    for k, v in values.items():
        print(f"{k}: frontier {v[0]:.2f}, local {v[1]:.2f}")
    for lang in TEXT:
        print(draw(lang, values))
