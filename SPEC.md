# Prompt for Claude Code: `calc-bounds` experiment repo

Paste everything below the line into Claude Code in an empty directory, or save it as `SPEC.md` in the repo root and tell Claude Code: "Read SPEC.md and start with Milestone 0."

---

## Context

I'm an emergency physician building a research experiment on clinical LLM systems. I want a clean, simple Python repo to test one idea:

**Separate what a model is good at from what code is good at.** A model extracts clinical parameters from a free-text note as typed, tri-state values with calibrated confidence. Deterministic code computes the clinical score, and also decides which missing parameters are *decision-relevant*: it only asks the clinician for a parameter if its value could change the score's decision category. We compare this against letting an LLM decide what to ask.

### Research question

Does separating calibrated tri-state extraction (model) from decision-relevance computation (code, via score bounds) reduce the number of questions asked without losing decision-category accuracy, compared with an end-to-end LLM agent?

### Hypotheses

- **H1:** The bounds policy asks far fewer questions than "ask for every missing field," with no loss in decision-category accuracy.
- **H2:** An end-to-end LLM agent both over-asks (irrelevant fields) and under-asks (commits while the category is still undetermined).
- **H3:** Tri-state typing (present / absent / unknown) reduces silent missing-treated-as-absent errors compared with a binary schema.
- **H4:** Echoing back low-confidence, decision-critical extracted values catches most extraction errors at a small question cost.

## Hard constraints

- **Synthetic data only.** No real patient data may ever enter this repo. Add `.gitignore` rules for `data/raw/`, `data/private/`, `.env`, and any cache directories.
- **Keep the architecture simple.** Plain Python, JSONL files and pandas. No databases, no web UI, no agent/orchestration frameworks (no LangChain, LlamaIndex, etc.). Thin provider clients only.
- **Do not implement clinical criteria from memory.** Every calculator must cite its primary source in the docstring. Where you are unsure of a criterion, threshold, point value or unit, implement your best reading, mark it `# TODO(physician-review)`, and list it in `docs/CALCULATOR_NOTES.md`. I will review these myself.
- **Verify, don't assume, package versions and APIs.** Check current versions of dependencies and current SDK interfaces before using them. Don't hardcode model names; put them in config.
- **Ask before adding dependencies** beyond this list: `pydantic` (v2), `pytest`, `hypothesis`, `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `pyyaml`, `typer`, `anthropic`, `openai` (used only as a client for OpenAI-compatible local servers such as vLLM or Ollama), `scikit-learn` (calibration only).
- **Reproducibility.** Every run is driven by a config file, fully seeded, and all LLM calls go through a disk cache keyed by a hash of (provider, model, prompt, parameters). Track token usage and cost per run.
- Use `uv` for environment and dependency management, Python 3.12, `ruff` for lint and format.

## Core data model (get this right first)

Each calculator parameter has two layers:

1. **Hidden truth:** the patient's actual value. Always defined; used only by the simulated clinician and for scoring.
2. **Documented state** in the note: `documented_positive` (value or finding stated), `documented_negative` (explicitly negated or stated normal), or `not_documented`.

The extractor's output per parameter is a tri-state type:

- `Present(value, unit, confidence, evidence_span)`
- `Absent(confidence, evidence_span)`
- `Unknown()`

Rules:

- `evidence_span` is character offsets into the note. The quoted text **must** be an exact substring of the note; reject and log any extraction whose span doesn't match.
- Units are explicit and normalized by code, never by the model. Build a small unit module with tested conversions (e.g., urea mmol/L ↔ mg/dL ↔ BUN mg/dL; creatinine mg/dL ↔ µmol/L; weight lb ↔ kg).
- **Unknown must never be treated as absent** anywhere in the scoring code, except in the explicit binary-schema ablation (S3-bin).

## Components

### 1. `calculators/`

Pure functions over typed parameters. Start with:

- HEART
- CURB-65 (note the urea/BUN unit issue)
- qSOFA
- PERC
- Wells criteria for PE (two-tier: PE likely / unlikely)
- Cockcroft-Gault, with configurable decision thresholds (e.g., dosing cutoffs)

Each calculator declares: its parameters and their domains (boolean, ordinal levels, or numeric with plausible physiological range), the score function, and its decision categories with thresholds.

### 2. `bounds/`

Given a partially known parameter set:

- Compute the set (or interval) of possible scores and the set of possible decision categories. Enumerate small discrete domains exactly; use interval arithmetic over physiological ranges for continuous ones.
- `is_determined(params) -> bool`: true when only one decision category is possible.
- `decision_relevant_missing(params) -> list[param]`: parameters whose resolution could change the set of possible categories.
- `voi_order(params, beliefs) -> list[param]`: given the model's probabilities for uncertain parameters, rank questions by the probability that each one alone resolves the category (greedy, one step).

**Property tests (hypothesis) are mandatory:** for any partial assignment, every completion's true score must fall within the computed bounds, and if `is_determined` is true, every completion must land in that same category.

### 3. `cohort/`

Structured synthetic patient generator:

- Samples hidden truth for each calculator's parameters from configurable, clinically plausible distributions.
- Assigns documented states with a configurable missingness rate.
- Guarantees coverage of both determined and undetermined cases.
- Tags cases with trap metadata: multiple encounters, negations, mixed units, contradictory values, comorbidity implied only by medication.
- Output: JSONL with the full ground truth.

### 4. `render/`

Turns a structured patient into a clinical note using an LLM:

- Parameters: language/locale (`en-US`, `es-CL`), a style guide file per locale, and the trap specifications.
- **Render each locale directly from the structure.** Never translate one rendered note into another language.
- A validation pass checks that the note is faithful to the structure: documented values appear, negations are expressed, and not-documented parameters aren't inferable from the text. Flag failures; don't silently drop them.
- An export command produces a random review sample (default 20%) as Markdown with the structure and note side by side, for physician review.

### 5. `extraction/`

Extractor interface returning the tri-state types above. Implementations:

- **Oracle extractor:** reads the ground truth directly. This is the upper bound, and it lets the whole pipeline run with zero API cost.
- **LLM extractor:** uses constrained or structured output. Use token logprobs for confidence where the provider exposes them; otherwise record the confidence as self-reported and flag it as such.
- **Calibration:** temperature scaling and isotonic regression, fitted on a dev split only.
- Provider clients: Anthropic API, plus an OpenAI-compatible client for local models.

### 6. `simulator/`

A deterministic simulated clinician that answers questions from the hidden truth. No LLM involved. It supports configurable "not available" answers (e.g., troponin not yet resulted) and logs every question asked.

### 7. `policies/`

- **S1 ask-all:** extract, then ask for every missing parameter.
- **S2 LLM agent:** a frontier LLM receives the note plus two tools, `calculate(calculator, params)` and `ask_clinician(question)`, and decides for itself what to ask and when to answer.
- **S3 bounds:** typed tri-state extraction, then code asks only for decision-relevant missing parameters, one at a time, recomputing after each answer.
- **S4:** S3 plus VOI ordering plus confidence echo (low-confidence, decision-critical extracted values are also confirmed).
- **S3-bin:** ablation of S3 with a binary schema (missing = absent).

Every policy must log a full trace: extractions, bounds at each step, questions asked, answers received, final score and category.

### 8. `eval/`

- **Primary metrics:** decision-category accuracy and questions per case; plot accuracy vs. questions per system.
- **Secondary metrics:** premature-commitment rate (committed while the category was undetermined), irrelevant-question rate, per-parameter extraction accuracy by documented state, calibration (Brier score, ECE, reliability diagrams), silent missing-as-absent errors, tokens, latency, cost.
- **Paired statistics:** McNemar for accuracy, Wilcoxon signed-rank for question counts, across systems on the same cases.
- **Error attribution:** trace every wrong final category to routing, extraction, bounds, or code.

### 9. `anchor/`

A loader that runs the S3 extraction-plus-code pipeline on MedCalc-Bench, for the calculators that overlap with ours, as a sanity check against published baselines. **Use the corrected, physician-adjudicated labels, not the original ones.** Find the current location and license of the corrected dataset and confirm them with me before downloading anything; don't guess URLs.

## Milestones

Work in this order. **Stop at the end of each milestone** and give me a summary: what was built, test status, anything marked for physician review, and open questions. Wait for my approval before continuing.

- **M0 — Plan and scaffold.** Propose the repo layout, config format, and module interfaces. Create `CLAUDE.md` with the conventions from this spec, `README.md`, `pyproject.toml`, the `.gitignore`, and an empty test suite that runs. Don't implement logic yet.
- **M1 — Calculators, units, bounds.** No LLM calls. Tests first for each calculator, using hand-worked examples from the primary sources, plus the property tests for bounds. Produce `docs/CALCULATOR_NOTES.md`.
- **M2 — Cohort, simulator, S1 and S3 with the oracle extractor.** This validates the bounds policy end to end at zero API cost and yields a first accuracy-vs-questions plot.
- **M3 — Renderer, validation pass, review export.** English only first.
- **M4 — LLM extractors, calibration, and the anchor run.**
- **M5 — S2 agent, S4, the S3-bin ablation, and the `es-CL` rendering of the same cohort.**
- **M6 — Full evaluation, statistics, plots, and a results summary.**

## Working style

- Small, focused commits with clear messages.
- Type hints everywhere; pydantic models at every module boundary.
- Prefer readable code over clever code. If you notice yourself over-engineering (plugin systems, abstract factories, deep class hierarchies), stop and simplify.
- When a decision is genuinely ambiguous (clinical or technical), ask me rather than guessing.

Start with **Milestone 0**.
