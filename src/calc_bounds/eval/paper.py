# ruff: noqa: RUF001  (en dashes are intentional in published CI ranges)
"""Tables and figures for Paper 1, regenerated from run outputs (no hand-computed numbers).

    uv run calc-bounds paper configs/main.yaml     -> paper/tables/, paper/figures/

Statistics: Wilson 95% CIs for proportions; percentile bootstrap (2,000 case resamples) for
mean questions; exact McNemar and Wilcoxon signed-rank for paired comparisons, Holm-adjusted
within each condition.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.eval import metrics
from calc_bounds.eval.plots import R_RC
from calc_bounds.eval.stats import paired_comparison
from calc_bounds.io import read_jsonl
from calc_bounds.policies import Trace

POLICY_ORDER = ["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]
POLICY_NAME = {
    "s1_ask_all": "Ask-all",
    "s2_llm_agent": "Agent",
    "s3_bounds": "Bounds",
    "s4_bounds_voi_echo": "Bounds + checks",
    "s3_bin": "Missing = normal",
}
CALC_NAME = {
    "heart": "HEART",
    "curb65": "CURB-65",
    "qsofa": "qSOFA",
    "perc": "PERC",
    "wells_pe": "Wells PE",
    "cockcroft_gault": "Cockcroft-Gault",
}
COMPARISONS = [
    ("s3_bounds", "s1_ask_all"),
    ("s3_bounds", "s2_llm_agent"),
    ("s3_bounds", "s3_bin"),
    ("s4_bounds_voi_echo", "s3_bounds"),
]


# --- statistics ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_mean(x: np.ndarray, seed: int = 0, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} ({100 * lo:.1f}–{100 * hi:.1f})"


# --- data -----------------------------------------------------------------------------------


def load_table(run: Path, label: str, cases: list[PatientCase], keep: set[str] | None = None):
    traces = read_jsonl(run / "traces" / f"{label}.jsonl", Trace)
    if keep is not None:
        traces = [t for t in traces if t.case_id in keep]
    return metrics.case_table(cases, traces)


def condition_rows(table: pd.DataFrame, condition: str) -> list[dict[str, object]]:
    rows = []
    for pol in POLICY_ORDER:
        g = table[table["policy"] == pol]
        if g.empty:
            continue
        n = len(g)
        q = g["n_questions"].to_numpy(dtype=float)
        qlo, qhi = boot_mean(q)
        asked = g["n_questions"].sum()
        committed_acc = g.loc[~g["abstained"], "correct"].mean()
        rows.append(
            {
                "Condition": condition,
                "System": POLICY_NAME[pol],
                "n": n,
                "Accuracy, % (95% CI)": pct(int(g["correct"].sum()), n),
                "Correct when answering, %": f"{100 * committed_acc:.1f}",
                "Under-triage, % (95% CI)": pct(int(g["under_triage"].sum()), n),
                "Over-triage, %": f"{100 * g['over_triage'].mean():.1f}",
                "Premature commitment, %": f"{100 * g['premature_commitment'].mean():.1f}",
                "Questions/case (95% CI)": f"{q.mean():.2f} ({qlo:.2f}–{qhi:.2f})",
                "Irrelevant questions, %": (
                    f"{100 * g['n_irrelevant_questions'].sum() / asked:.1f}" if asked else "–"
                ),
            }
        )
    return rows


def comparison_rows(table: pd.DataFrame, condition: str) -> list[dict[str, object]]:
    comps = [paired_comparison(table, a, b) for a, b in COMPARISONS]
    p_acc = holm([c["mcnemar_p"] for c in comps])
    p_q = holm([c["wilcoxon_p"] for c in comps])
    return [
        {
            "Condition": condition,
            "Comparison": f"{POLICY_NAME[c['a']]} vs {POLICY_NAME[c['b']]}",
            "Accuracy A vs B, %": f"{100 * c['acc_a']:.1f} vs {100 * c['acc_b']:.1f}",
            "Discordant (A only / B only)": f"{c['only_a_correct']} / {c['only_b_correct']}",
            "McNemar p (Holm)": fmt_p(pa),
            "Questions A vs B": f"{c['mean_q_a']:.2f} vs {c['mean_q_b']:.2f}",
            "Wilcoxon p (Holm)": fmt_p(pq),
        }
        for c, pa, pq in zip(comps, p_acc, p_q, strict=True)
    ]


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


SHORT_NAME = POLICY_NAME


def summary_table(ideal: pd.DataFrame, noisy: pd.DataFrame) -> pd.DataFrame:
    """The paper's single results table: measures as rows, policies as columns (Haiku
    extraction, clean notes, full cohort; ideal and noisy clinician)."""

    def per_policy(t: pd.DataFrame, fn) -> dict[str, str]:
        return {SHORT_NAME[p]: fn(t[t["policy"] == p]) for p in POLICY_ORDER}

    def rate(col: str):
        return lambda g: f"{100 * g[col].mean():.1f}"

    def irrelevant(g: pd.DataFrame) -> str:
        asked = g["n_questions"].sum()
        return f"{100 * g['n_irrelevant_questions'].sum() / asked:.1f}" if asked else "–"

    rows = [
        ("Ideal", "Accuracy, %", rate("correct")),
        ("Ideal", "Questions per case", lambda g: f"{g['n_questions'].mean():.2f}"),
        ("Ideal", "Irrelevant questions, %", irrelevant),
        ("Ideal", "Answered too early, %", rate("premature_commitment")),
        ("Ideal", "Under-triage, %", rate("under_triage")),
        ("Noisy", "Accuracy, %", rate("correct")),
        ("Noisy", "Answered too early, %", rate("premature_commitment")),
        ("Noisy", "Under-triage, %", rate("under_triage")),
    ]
    out, previous = [], None
    for clin, measure, fn in rows:
        t = ideal if clin == "Ideal" else noisy
        label = clin if clin != previous else ""
        previous = clin
        out.append({"Clinician": label, "Measure": measure} | per_policy(t, fn))
    return pd.DataFrame(out)


def cohort_table(cases: list[PatientCase], calcs: dict) -> pd.DataFrame:
    rows = []
    for cid, name in CALC_NAME.items():
        cs = [c for c in cases if c.calculator == cid]
        if not cs:
            continue
        cats = pd.Series([c.true_category for c in cs]).value_counts()
        order = calcs[cid].category_names()
        missing = np.mean(
            [np.mean([s.value == "not_documented" for s in c.documented.values()]) for c in cs]
        )
        rows.append(
            {
                "Calculator": name,
                "Cases": len(cs),
                "Parameters": len(calcs[cid].parameters),
                "True categories": ", ".join(f"{k} {int(cats.get(k, 0))}" for k in order),
                "Undetermined from note": sum(not c.determined_from_note for c in cs),
                "Params not documented, %": f"{100 * missing:.0f}",
                "Cases with traps": sum(bool(c.traps) for c in cs),
            }
        )
    return pd.DataFrame(rows)


def anchor_table(run: Path) -> pd.DataFrame:
    rows = []
    for split, label in [("train", "__train"), ("test", "")]:
        for x, xname in [("haiku", "Haiku 4.5"), ("qwen_local", "Qwen3.5-9B")]:
            path = run / "anchor" / f"{x}{label}.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path)
            for cid, g in [*list(d.groupby("calculator")), ("all", d)]:
                n = len(g)
                rows.append(
                    {
                        "Split": split,
                        "Extractor": xname,
                        "Calculator": CALC_NAME.get(cid, "All"),
                        "n": n,
                        "Code reproduces label, %": f"{100 * g['impl_agrees'].mean():.0f}",
                        "Determined from note, % (95% CI)": pct(
                            int(g["tristate_determined"].sum()), n
                        ),
                        "Truth still possible, %": f"{100 * g['tristate_gt_possible'].mean():.0f}",
                        "Category correct, missing=normal, %": (
                            f"{100 * g['medcalc_convention_category_correct'].mean():.0f}"
                        ),
                        "Entity agreement, %": f"{100 * g['entity_agreement'].mean():.0f}",
                    }
                )
    return pd.DataFrame(rows)


def compact_tables(full: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Narrow versions for the manuscript body (full tables go to the supplement)."""
    t2 = full["table2_main"][
        [
            "Condition",
            "System",
            "Accuracy, % (95% CI)",
            "Questions/case (95% CI)",
            "Irrelevant questions, %",
            "Premature commitment, %",
            "Under-triage, % (95% CI)",
        ]
    ].rename(
        columns={
            "Condition": "Extraction",
            "Irrelevant questions, %": "Irrelevant, %",
            "Premature commitment, %": "Premature, %",
        }
    )
    t3 = full["table3_conditions"]
    t3 = t3[t3["Condition"].str.startswith("Haiku")][
        [
            "Condition",
            "System",
            "Accuracy, % (95% CI)",
            "Correct when answering, %",
            "Under-triage, % (95% CI)",
            "Over-triage, %",
            "Premature commitment, %",
            "Questions/case (95% CI)",
        ]
    ].copy()
    t3["Condition"] = t3["Condition"].str.replace("Haiku, ", "", regex=False)
    t3 = t3.rename(
        columns={
            "Correct when answering, %": "Correct if answered, %",
            "Premature commitment, %": "Premature, %",
            "Questions/case (95% CI)": "Questions/case",
        }
    )
    t4 = full["table6_real_notes"]
    t4 = t4[t4["Split"] == "train"][
        [
            "Extractor",
            "Calculator",
            "n",
            "Code reproduces label, %",
            "Determined from note, % (95% CI)",
            "Truth still possible, %",
            "Category correct, missing=normal, %",
        ]
    ]
    t2["Extraction"] = t2["Extraction"].str.replace(" extraction", "", regex=False)
    t3["Condition"] = (
        t3["Condition"]
        .str.replace(" notes, ", " / ", regex=False)
        .str.replace(" clinician", "", regex=False)
    )
    return {"table2_compact": t2, "table3_compact": t3, "table4_compact": t4}


HEART_RISK_FACTORS = [
    "hypertension",
    "hypercholesterolemia",
    "diabetes",
    "obesity",
    "smoking",
    "family_history_cad",
]


def implausible_cases(cases: list[PatientCase]) -> dict[str, str]:
    """Cases whose independently sampled characteristics look clinically implausible."""
    out: dict[str, str] = {}
    for c in cases:
        if c.calculator != "heart":
            continue
        n_rf = sum(bool(c.truth[r]) for r in HEART_RISK_FACTORS)
        if c.truth["atherosclerotic_disease"] and n_rf == 0:
            out[c.case_id] = "established atherosclerotic disease with no risk factors"
        elif c.truth["age"] < 40 and (n_rf >= 3 or c.truth["atherosclerotic_disease"]):
            out[c.case_id] = "age < 40 with >= 3 risk factors or atherosclerotic disease"
    return out


def implausible_sensitivity(run: Path, cases: list[PatientCase]) -> pd.DataFrame:
    flags = implausible_cases(cases)
    rows = []
    for label, name in [
        ("oracle", "Oracle"),
        ("haiku__sonnet", "Haiku 4.5"),
        ("qwen_local__sonnet", "Qwen3.5-9B"),
    ]:
        t = load_table(run, label, cases)
        for subset, g in [("all", t), ("excluding flagged", t[~t["case_id"].isin(flags)])]:
            for pol in POLICY_ORDER:
                h = g[g["policy"] == pol]
                rows.append(
                    {
                        "Extraction": name,
                        "Cases": subset,
                        "System": POLICY_NAME[pol],
                        "n": len(h),
                        "Accuracy, %": f"{100 * h['correct'].mean():.1f}",
                        "Under-triage, %": f"{100 * h['under_triage'].mean():.1f}",
                        "Questions/case": f"{h['n_questions'].mean():.2f}",
                    }
                )
    df = pd.DataFrame(rows)
    df.attrs["n_flagged"] = len(flags)
    return df


# --- figures --------------------------------------------------------------------------------


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


CONDITIONS = {  # label -> (table key, marker, filled): circles clean, triangles messy notes
    "Clean notes, ideal clinician": ("clean/ideal", "o", True),
    "Messy notes, ideal clinician": ("messy/ideal", "^", True),
    "Clean notes, noisy clinician": ("clean/noisy", "o", False),
    "Messy notes, noisy clinician": ("messy/noisy", "^", False),
}


def _dotchart(tables: dict[str, pd.DataFrame], panels, out: Path, name: str) -> None:
    """Base-R `dotchart` layout: policies as rows, one panel per measure, one symbol per
    condition (filled: ideal clinician; open: noisy; circle: clean notes; triangle: messy)."""
    pols = list(reversed(POLICY_ORDER))  # dotchart draws the first row at the top
    fig, axes = plt.subplots(1, len(panels), figsize=(9, 3.4), sharey=True)
    for ax, (xlabel, measure, xlim, xticks) in zip(axes, panels, strict=True):
        for y in range(len(pols)):
            ax.axhline(y, color="gray", linestyle=":", linewidth=0.7, zorder=0)
        for label, (key, marker, filled) in CONDITIONS.items():
            t = tables[key]
            x = [measure(t[t.policy == p]) for p in pols]
            ax.scatter(
                x,
                range(len(pols)),
                marker=marker,
                s=36,
                linewidths=1.1,
                edgecolors="black",
                facecolors="black" if filled else "white",
                zorder=3,
                label=label if ax is axes[0] else None,
            )
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.6, len(pols) - 0.4)
    axes[0].set_yticks(range(len(pols)), [SHORT_NAME[p] for p in pols])
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.legend(fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig_tradeoff(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Accuracy and questions per case (Haiku extraction, 559 paired cases)."""
    _dotchart(
        tables,
        [
            (
                "Decision-category accuracy (%)",
                lambda g: 100 * g["correct"].mean(),
                (80, 101),
                range(80, 101, 5),
            ),
            (
                "Questions per case",
                lambda g: g["n_questions"].mean(),
                (0, 2.2),
                [0, 0.5, 1, 1.5, 2],
            ),
        ],
        out,
        "fig2_accuracy_vs_questions",
    )


def fig_safety(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Under- and over-triage with the higher-risk fallback (Haiku extraction, 559 cases)."""
    _dotchart(
        tables,
        [
            (
                "Under-triage (% of cases)",
                lambda g: 100 * g["under_triage"].mean(),
                (-0.5, 15),
                range(0, 16, 5),
            ),
            (
                "Over-triage (% of cases)",
                lambda g: 100 * g["over_triage"].mean(),
                (-0.5, 15),
                range(0, 16, 5),
            ),
        ],
        out,
        "fig3_safety_triage",
    )


def fig_real_notes(run: Path, out: Path) -> None:
    """Dot chart: per calculator, share of real case reports whose category the note determines,
    and share whose category is correct when missing inputs are treated as normal."""
    d = pd.read_csv(run / "anchor" / "haiku__train.csv")
    order = ["heart", "wells_pe", "curb65", "cockcroft_gault", "perc"]  # bottom to top
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for y in range(len(order)):
        ax.axhline(y, color="gray", linestyle=":", linewidth=0.7, zorder=0)
    for col, label, face in [
        ("tristate_determined", "Category determined from the note", "black"),
        ("medcalc_convention_category_correct", "Category correct if missing = normal", "white"),
    ]:
        x = [100 * d.loc[d.calculator == c, col].mean() for c in order]
        ax.scatter(
            x,
            range(len(order)),
            marker="o",
            s=36,
            linewidths=1.1,
            edgecolors="black",
            facecolors=face,
            zorder=3,
            label=label,
        )
    n = [int((d.calculator == c).sum()) for c in order]
    ax.set_yticks(
        range(len(order)), [f"{CALC_NAME[c]} (n = {k})" for c, k in zip(order, n, strict=True)]
    )
    ax.set_xlim(0, 101)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Real case reports (%)")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.legend(fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_real_notes.{ext}", dpi=300)
    plt.close(fig)


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    steps = [
        "Clinical note",
        "LLM extraction\n(present / absent /\nunknown, evidence,\nconfidence)",
        "Code: units,\ncalculator, score\nbounds over unknowns",
        "Determined?\nIf not, ask only\ndecision-relevant\nitems",
        "Decision category\n(or higher-risk\nfallback)",
    ]
    w, gap = 0.16, 0.045
    xs = [0.01 + i * (w + gap) for i in range(len(steps))]
    for x, text in zip(xs, steps, strict=True):
        ax.add_patch(
            plt.Rectangle((x, 0.18), w, 0.56, facecolor="white", edgecolor="black", linewidth=0.8)
        )
        ax.text(x + w / 2, 0.46, text, ha="center", va="center", fontsize=8)
    for x in xs[:-1]:
        ax.annotate(
            "",
            xy=(x + w + gap - 0.004, 0.46),
            xytext=(x + w + 0.004, 0.46),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
        )
    ax.annotate(
        "",
        xy=(xs[2] + w / 2, 0.76),
        xytext=(xs[3] + w / 2, 0.76),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 0.8,
            "linestyle": "--",
            "connectionstyle": "arc3,rad=0.35",
        },
    )
    ax.text(
        (xs[2] + xs[3] + w) / 2,
        0.97,
        "clinician's answer updates the bounds",
        ha="center",
        fontsize=7.5,
        style="italic",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.02)
    _save(fig, out, "fig1_pipeline")


def supplement_tables(run: Path, calcs: dict, cases: list[PatientCase]) -> dict[str, pd.DataFrame]:
    from calc_bounds.cohort.priors import DEFAULT_PRIORS
    from calc_bounds.distributions import Bernoulli, Categorical, TruncNormal

    rows = []
    for cid, name in CALC_NAME.items():
        for pid, d in DEFAULT_PRIORS[cid].items():
            match d:
                case Bernoulli(p=p):
                    desc = f"P(yes) = {p:g}"
                case Categorical(probs=pr):
                    levels = calcs[cid].param(pid).domain.levels  # type: ignore[union-attr]
                    desc = ", ".join(f"{lv} {q:g}" for lv, q in zip(levels, pr, strict=True))
                case TruncNormal():
                    desc = f"Normal({d.mean:g}, {d.sd:g}) truncated to [{d.lo:g}, {d.hi:g}]"
            rows.append({"Calculator": name, "Parameter": pid, "Prior": desc})
    report = run / "report"
    miss = pd.read_csv(report / "missingness_sensitivity.csv")
    miss["policy"] = miss["policy"].map(POLICY_NAME)
    miss = miss.rename(
        columns={
            "missingness": "Missingness",
            "policy": "System",
            "accuracy_natural": "Accuracy (natural share)",
            "questions_natural": "Questions/case (natural share)",
            "mean_natural_undetermined": "Undetermined share",
        }
    ).round(3)
    sweep = pd.read_csv(report / "s4_echo_threshold_sweep.csv")
    sweep = sweep[~sweep["extraction"].str.contains("noisy")].round(4)
    sweep["extraction"] = sweep["extraction"].map(
        {"haiku__sonnet": "Haiku 4.5", "qwen_local__sonnet": "Qwen3.5-9B"}
    )
    sweep = sweep.rename(
        columns={
            "extraction": "Extractor",
            "echo_threshold": "Confirm below confidence",
            "accuracy": "Accuracy",
            "mean_questions": "Questions/case",
            "echo_per_case": "Confirmations/case",
            "echo_caught_errors": "Errors caught",
        }
    )
    return {"s2_priors": pd.DataFrame(rows), "s3_missingness": miss, "s4_echo_sweep": sweep}


def prompt_examples(cfg_calcs: dict, cases: list[PatientCase], out: Path) -> None:
    """Exact prompts for one example case (supplement S7)."""
    from calc_bounds.extraction.llm import EXTRACT_SYSTEM, extraction_prompt
    from calc_bounds.policies.llm_agent import AGENT_SYSTEM, initial_prompt
    from calc_bounds.render.prompts import JUDGE_SYSTEM, RENDER_SYSTEM, judge_prompt, render_prompt

    case = next(c for c in cases if c.calculator == "curb65" and c.traps)
    calc = cfg_calcs[case.calculator]
    note = "<the rendered note>"
    parts = [
        ("Renderer (system)", RENDER_SYSTEM),
        ("Renderer (user), example CURB-65 case", render_prompt(case, calc, "en-US")),
        ("Judge (system)", JUDGE_SYSTEM),
        ("Judge (user)", judge_prompt(note, calc)),
        ("Extractor (system)", EXTRACT_SYSTEM),
        ("Extractor (user)", extraction_prompt(note, calc)),
        ("Agent (system)", AGENT_SYSTEM),
        ("Agent (first user turn)", initial_prompt(note, calc)),
    ]
    text = "\n\n".join(f"### {h}\n\n```text\n{b}\n```" for h, b in parts)
    (out / "s7_prompts.md").write_text(text + "\n")


# --- entry point ----------------------------------------------------------------------------


def build(run: Path, out: Path, calcs: dict) -> dict[str, object]:
    tables_dir, figs = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(run / "cohort.jsonl", PatientCase)
    messy_ok = {
        json.loads(line)["case_id"]
        for line in (run / "notes" / "messy.jsonl").read_text().splitlines()
        if line.strip() and not json.loads(line)["validation_failed"]
    }

    # Table 1: cohort.
    cohort = cohort_table(cases, calcs)
    # Table 2: main results, clean notes, ideal clinician, all extraction sources (n=1,200).
    main_rows = []
    for label, name in [
        ("oracle", "Oracle extraction"),
        ("haiku__sonnet", "Haiku 4.5"),
        ("qwen_local__sonnet", "Qwen3.5-9B (local)"),
    ]:
        main_rows += condition_rows(load_table(run, label, cases), name)
    # Table 3: harder conditions, paired on the 559 cases with validated messy notes.
    hard_rows, comp_rows, fig_tables = [], [], {}
    for x, xname in [("haiku", "Haiku"), ("qwen_local", "Qwen")]:
        for notes, nname in [("sonnet", "clean"), ("messy", "messy")]:
            for clin, cname in [("", "ideal"), ("__noisy", "noisy")]:
                label = f"{x}__{notes}{clin}"
                t = load_table(run, label, cases, keep=messy_ok)
                cond = f"{xname}, {nname} notes, {cname} clinician"
                hard_rows += condition_rows(t, cond)
                comp_rows += comparison_rows(t, cond)
                if x == "haiku":
                    fig_tables[f"{nname}/{cname}"] = t
    # Clean-notes comparisons on the full 1,200.
    full_comp = []
    for label, name in [
        ("oracle", "Oracle, clean, ideal"),
        ("haiku__sonnet", "Haiku, clean, ideal"),
        ("qwen_local__sonnet", "Qwen, clean, ideal"),
        ("haiku__sonnet__noisy", "Haiku, clean, noisy"),
    ]:
        full_comp += comparison_rows(load_table(run, label, cases), name)

    outputs = {
        "table_summary": summary_table(
            load_table(run, "haiku__sonnet", cases), load_table(run, "haiku__sonnet__noisy", cases)
        ),
        "table1_cohort": cohort,
        "table2_main": pd.DataFrame(main_rows),
        "table3_conditions": pd.DataFrame(hard_rows),
        "table4_comparisons_full": pd.DataFrame(full_comp),
        "table5_comparisons_paired559": pd.DataFrame(comp_rows),
        "table6_real_notes": anchor_table(run),
    }
    outputs |= compact_tables(outputs)
    outputs |= supplement_tables(run, calcs, cases)
    outputs["s10_implausible_sensitivity"] = implausible_sensitivity(run, cases)
    flags = implausible_cases(cases)
    (tables_dir / "s10_flagged_cases.md").write_text(
        f"{len(flags)} of {len(cases)} cases flagged: "
        + "; ".join(f"{k} ({v})" for k, v in sorted(flags.items()))
        + "\n"
    )
    prompt_examples(calcs, cases, tables_dir)
    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)
        (tables_dir / f"{name}.md").write_text(to_markdown(df))

    matplotlib.rcParams.update(R_RC)
    fig_pipeline(figs)
    fig_tradeoff(fig_tables, figs)
    fig_safety(fig_tables, figs)
    fig_real_notes(run, figs)
    import shutil

    shutil.copy(run / "report" / "reliability.png", figs / "figS5_reliability.png")
    shutil.copy(run / "report" / "reliability.pdf", figs / "figS5_reliability.pdf")
    return {"tables": sorted(outputs), "messy_validated_cases": len(messy_ok)}
