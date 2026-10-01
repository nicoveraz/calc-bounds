"""LinkedIn figure (1200 x 1500 px, 4:5 portrait), in English and Spanish.

Numbers come from paper/tables/table_summary.csv (written by `calc-bounds paper`); nothing is
hand-entered. Style follows the paper figures (base-R look). Output: paper/social/.

    uv run python scripts/social_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calc_bounds.eval.plots import R_RC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "social"
ALARM = "#DF536B"  # R 4 palette red: missing = normal
OURS = "#2297E6"  # R 4 palette blue: bounds

TEXT = {
    "en": {
        "title": "Unknown is not normal",
        "subtitle": "When a clinical note does not mention a finding,\n"
        "the common convention is to assume it is normal.",
        "names": {
            "Missing = normal": "Assume missing\n= normal",
            "Ask-all": "Ask about\neverything missing",
            "Agent": "AI agent alone\n(Claude Opus 5.5)",
            "Bounds": "AI reads, code decides,\nasks only what matters",
        },
        "p1": "Patients placed in a lower-risk group than their true one",
        "p1_note": "about 1 in 12",
        "p2": "Questions to the clinician per patient",
        "p2_note": "half of asking about everything, same accuracy",
        "unit1": "%",
        "foot": "Simulation: 1,200 synthetic emergency cases, about 30% of findings unmentioned,\n"
        "Claude Haiku 4.5 reading the notes, clinician always answering correctly.",
        "cite": "Vera Zúñiga N. arXiv:2609.34112 · github.com/nicoveraz/calc-bounds",
    },
    "es": {
        "title": "Desconocido no es normal",
        "subtitle": "Cuando una nota clínica no menciona un hallazgo,\n"
        "la convención habitual es asumir que es normal.",
        "names": {
            "Missing = normal": "Asumir faltante\n= normal",
            "Ask-all": "Preguntar por\ntodo lo faltante",
            "Agent": "Agente de IA solo\n(Claude Opus 5.5)",
            "Bounds": "La IA lee, el código decide,\npregunta solo lo necesario",
        },
        "p1": "Pacientes clasificados en un grupo de menor riesgo que el real",
        "p1_note": "cerca de 1 de cada 12",
        "p2": "Preguntas al clínico por paciente",
        "p2_note": "la mitad que preguntar por todo, misma precisión",
        "unit1": " %",
        "foot": "Simulación: 1.200 casos sintéticos de urgencia, cerca del 30 % de los "
        "hallazgos sin\nmencionar, Claude Haiku 4.5 leyendo las notas, clínico que siempre "
        "responde bien.",
        "cite": "Vera Zúñiga N. arXiv:2609.34112 · github.com/nicoveraz/calc-bounds",
    },
}


def load() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "paper" / "tables" / "table_summary.csv")
    ideal = t.iloc[: t["Clinician"].ffill().eq("Ideal").sum()]
    return ideal.set_index("Measure")


def fmt(v: float, lang: str, decimals: int) -> str:
    s = f"{v:.{decimals}f}"
    return s.replace(".", ",") if lang == "es" else s


def panel(ax, values: dict[str, float], names, lang: str, decimals: int, unit: str, xmax: float):
    order = list(names)
    y = range(len(order))[::-1]
    colors = [
        ALARM if p == "Missing = normal" else OURS if p == "Bounds" else "#9E9E9E" for p in order
    ]
    vals = [values[p] for p in order]
    ax.barh(list(y), vals, height=0.62, color=colors, edgecolor="black", linewidth=0.8)
    for yi, v in zip(y, vals, strict=True):
        ax.text(
            v + xmax * 0.015,
            yi,
            fmt(v, lang, decimals) + unit,
            va="center",
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
    ax.set_yticks(list(y), [names[p] for p in order], fontsize=13.5)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)


def draw(lang: str, df: pd.DataFrame) -> Path:
    tx = TEXT[lang]
    under = {p: float(df.loc["Under-triage, %", p]) for p in tx["names"]}
    questions = {p: float(df.loc["Questions per case", p]) for p in tx["names"]}
    with matplotlib.rc_context(R_RC):
        fig = plt.figure(figsize=(8, 10), dpi=150, facecolor="white")  # 1200 x 1500 px
        fig.text(0.06, 0.945, tx["title"], fontsize=34, fontweight="bold", va="top")
        fig.text(0.06, 0.875, tx["subtitle"], fontsize=16, va="top", color="#333333")

        ax1 = fig.add_axes((0.40, 0.50, 0.52, 0.235))
        fig.text(0.06, 0.775, tx["p1"], fontsize=16, fontweight="bold", va="top")
        fig.text(0.06, 0.748, tx["p1_note"], fontsize=14, va="top", color=ALARM, fontweight="bold")
        panel(ax1, under, tx["names"], lang, 1, tx["unit1"], 11)

        ax2 = fig.add_axes((0.40, 0.145, 0.52, 0.235))
        fig.text(0.06, 0.44, tx["p2"], fontsize=16, fontweight="bold", va="top")
        fig.text(0.06, 0.413, tx["p2_note"], fontsize=14, va="top", color=OURS, fontweight="bold")
        panel(ax2, questions, tx["names"], lang, 2, "", 2.3)

        fig.add_artist(plt.Line2D([0.06, 0.94], [0.115, 0.115], color="black", linewidth=0.8))
        fig.text(0.06, 0.098, tx["foot"], fontsize=11, va="top", color="#333333")
        fig.text(0.06, 0.04, tx["cite"], fontsize=11.5, va="top", fontweight="bold")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"linkedin_{lang}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


if __name__ == "__main__":
    data = load()
    for lang in TEXT:
        print(draw(lang, data))
