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

## Config format
YAML validated by `config.RunConfig` (extra keys rejected). See
`configs/m2_oracle.yaml`. Sections: `run_id`, `seed`, `calculators`,
`calculator_options`, `cohort`, `render`, `extraction`, `simulator`, `policies`, `providers`
(model names, base URLs, env var names for keys, prices).

## CLI
`calc-bounds check-config | cohort | render | export-review | run | eval | anchor  CONFIG`
