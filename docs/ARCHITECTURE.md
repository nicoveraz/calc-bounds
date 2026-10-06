# Architecture (M0 proposal)

## Layout
```
configs/                 run configs (YAML); one per run
data/                    synthetic cohorts only; data/raw/ and data/private/ are gitignored
docs/                    ARCHITECTURE.md, CALCULATOR_NOTES.md (M1)
runs/<run_id>/           outputs (gitignored): cohort.jsonl, notes.jsonl, traces.jsonl, metrics, plots
src/calc_bounds/
  types.py               core data model: domains, ParameterSpec, DocumentedState, tri-state Extraction
  units.py               unit normalization (code only)
  config.py              RunConfig (pydantic) + load_config
  cli.py                 typer entry point `calc-bounds`
  calculators/           base.py (Calculator, DecisionCategory) + one module per calculator; REGISTRY
  bounds/                score_bounds, is_determined, decision_relevant_missing, voi_order
  cohort/                PatientCase, Trap; generate_cohort
  render/                render_note, validate_note, export_review_sample; styles/<locale>.md
  extraction/            Extractor protocol, oracle (M2), llm (M4), calibration
  llm/                   LLMRequest/Response, thin Anthropic + OpenAI-compatible clients, disk cache, usage
  simulator/             SimulatedClinician (deterministic, from hidden truth)
  policies/              Trace/Step models; S1, S2, S3, S4, S3-bin
  eval/                  metrics, paired stats, error attribution, plots
  anchor/                MedCalc-Bench loader (M4, after dataset confirmation)
tests/
```

## Data flow
```
config ──► cohort.generate ──► cohort.jsonl (PatientCase: truth + documented state + traps)
                                   │
                       render (per locale, from structure) ──► notes.jsonl (+ validation issues)
                                   │
          policy.run(case, note, calc, extractor, clinician) ──► traces.jsonl
                                   │
                              eval ──► metrics.csv, stats, plots
```
The oracle extractor lets `cohort → policy → eval` run with no notes and no API cost (M2).

## Key design choices
1. **Shared parameter catalog.** `ParameterSpec`s are defined once (age, SBP, RR, ...) and
   referenced by calculators, so hidden truth, extraction and questions are per-parameter, not
   per-calculator.
2. **Extract raw values, derive criteria in code.** For numeric criteria (e.g. "HR > 100") the
   extractor returns the measured value; the calculator applies thresholds. Keeps units and
   thresholds out of the model.
3. **What Absent means is declared per parameter.** Bool → False; ordinal → level 0; numeric →
   `absent_means` interval ("stated normal"), or not allowed (`None`).
4. **Bounds input is a partial assignment** `{param: Exact(value) | Interval(lo, hi)}`; params
   not in the map are Unknown and range over their full domain. Converting extractions to a
   partial assignment is the only place tri-state semantics enter bounds, which makes S3-bin a
   one-line switch (Unknown → Absent) that is easy to audit.
5. **Exact vs interval bounds.** Bool/ordinal enumerated. Numeric step-function params are
   enumerated over the regions between declared `breakpoints` (exact). Continuous scores
   (Cockcroft-Gault) use interval corners, exact when the calculator declares `monotone`.
   A missing param is *decision-relevant* if, for some completion of the other unknowns,
   changing it changes the category (so params that only matter jointly still count). This is
   exact for point scores; for Cockcroft-Gault it is conservative (all unknowns while
   undetermined). Hypothesis property tests check soundness: every completion's score lies in the bounds; if
   determined, every completion has the same category.
6. **One case = one calculator** (a patient case is generated for a target calculator).
7. **LLM plumbing** is a Protocol with two thin clients; every call goes through
   `llm.cache_key` → disk cache (`.cache/llm/<sha256>.json`). Prices per model come from config;
   `max_cost_usd` is a hard budget per run (0 = no paid calls).

## M2 pipeline decisions
- **Cohort coverage:** rejection sampling fills an exact quota of cases whose category is
  undetermined from the note (`undetermined_fraction`). Seeds are derived from
  (run seed, calculator id) and (run seed, case id), so outputs don't depend on list order.
- **Documented state:** false booleans are documented as negated. Normal numeric/ordinal
  values are written as "normal" with probability `normal_as_negation_rate`; otherwise the
  value is stated. Traps are rendering instructions only and never change truth.
- **Abstention:** if a relevant answer is "not available" and nothing else can settle the
  category, the policy abstains (`final_category = None`). Code policies never guess. **Primary outcome
  (decided with the user): abstention counts as incorrect.** Secondary: `coverage` and
  `accuracy_when_committed`.
- **Silent missing-as-absent:** a trace records `initial_known`, the constraints before any
  question. A constraint on a not-documented param is counted as a silent error.

## M3 rendering and validation
- **Fact sheet.** Each case becomes per-parameter instructions:
  - STATE (with the exact number to write)
  - STATE AS NORMAL (no number)
  - DO NOT MENTION (nothing inferable)
  - plus trap instructions: an embedded negation, another unit, a dated prior value, a
    superseded triage value, or a comorbidity shown only through a medication.
  The note must not name any score or rule.
- **Locales** are rendered directly from the structure; the fact sheet is language-neutral
  English, and style guides live in `render/styles/<locale>.md`.
- **Validation** has two layers. Failures are recorded as issues and never dropped.
  - Rule checks (deterministic): every required number appears in its exact surface form,
    not-documented numeric values don't appear (warning), no score names, minimum length.
  - A judge model (must differ from the renderer) classifies each parameter as stated,
    negated/normal, not mentioned or implied, and gives the ordinal level. Any mismatch
    with the documented state is an error. "Implied" catches leaks.
- **Session provider.** `render`/`judge` export cache misses to `runs/<run>/pending/`. They
  are answered offline (Claude Code subagents on the configured model) and loaded with
  `import-responses`, which checks each response against its pending request hash. Re-running
  the stage then reads from the cache. Notes are replayable from the cache but not
  regenerable byte-for-byte.
- **Retries (rejection sampling for faithfulness).** A note that fails validation (any rule
  or judge error) is re-rendered, up to `max_attempts` per render set (default 3). Attempt n > 1
  adds `attempt: n` to the request params, so it gets its own cache key; local models also get
  a different seed. `render` walks the attempts from the cache and keeps the first one that
  passes. Every note records `attempt`, `prompt_version` and `validation_failed` (the last
  attempt still fails). Retry rates are reported; notes that still fail are excluded from the
  primary analysis, never silently dropped.
- **Prompt versions.** `render/prompts.PROMPT_VERSION` is bumped on any wording change, and a
  whole cohort is rendered under one version. v3 followed the 300-note stage 1 (5% judge
  failures on v2: heart-rate leaks through rhythm descriptors, omitted negations, HEART history
  leaking through pain characterisation).
- **Renderer-family bias.** `renders` holds named render sets over the same cases (e.g.
  `sonnet` for all cases, `local` for a 25-per-calculator subset) for the 2×2 renderer ×
  extractor-family comparison in M4.

## M4 extraction and calibration ($0 plan)
- **Providers.**
  - `claude_cli`: Claude via headless Claude Code (`claude -p`), billed to the subscription.
    The flags make it as close to a plain model call as the CLI allows: our own
    `--system-prompt`, `--tools ""`, `--strict-mcp-config`,
    `--exclude-dynamic-system-prompt-sections`, `--no-session-persistence`, `--json-schema`,
    run from an empty directory. About 1.2k input tokens of harness overhead per call.
    Methods wording: "Claude via Claude Code headless".
  - `ollama`: the native `/api/chat` endpoint, which supports JSON schema, `think`, seed and
    token logprobs. The OpenAI-compatible endpoint has no logprobs.
- **Extractor output.** Per parameter: status, value, unit token, exact evidence quote, and a
  self-reported confidence. Code then:
  - locates the quote (non-exact quotes are rejected and the parameter becomes Unknown);
  - normalizes units;
  - rejects Absent for non-negatable parameters and values outside the plausible range.
- **Confidence.** With logprobs: P(status tokens) × P(value tokens) of that parameter's
  object. Without them: self-reported, and flagged as such.
- **Scoring a claim.** Present is correct if the item is documented positive and the value
  matches the truth (numeric in canonical units). Absent is correct if the item is documented
  negative. Unknown is correct if the item is not documented. Level 0 of a negatable graded
  item counts as normal.
- **Calibration.** None, temperature scaling or isotonic, fitted on a seeded 30% dev split per
  calculator. Brier, ECE and reliability are reported on the remaining test split.
- **Outputs.**
  - `extractions/<extractor>__<render>.jsonl`
  - `calibration/<extractor>__<render>.json`
  - `traces/<label>.jsonl` and `eval/<label>/`, where the label is `oracle` or
    `<extractor>__<render>`.

## Config format
YAML validated by `config.RunConfig` (extra keys rejected). See
`configs/m2_oracle.yaml`. Sections: `run_id`, `seed`, `calculators`,
`calculator_options`, `cohort`, `render`, `extraction`, `simulator`, `policies`, `providers`
(model names, base URLs, env var names for keys, prices).

## CLI
`calc-bounds check-config | cohort | render | export-review | run | eval | anchor  CONFIG`

## MIMIC-IV validation (code only)
`src/calc_bounds/mimic/` runs the same extraction, bounds and S1/S3/S4/S3-bin policies on real
records, with the EHR as the clinician. Data, row-level outputs and the LLM cache must live
outside the repo and only local models are accepted (enforced by `mimic/config.py`). Only
aggregate tables are meant to be committed. Design, outcomes and review items:
[`MIMIC_VALIDATION.md`](MIMIC_VALIDATION.md).

`calc-bounds mimic check-config | cohort | annotation-template | import-annotations | run | aggregate  CONFIG`
