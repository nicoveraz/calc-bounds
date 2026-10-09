# Validation on MIMIC-IV (real, retrospective records)

Status: **code only, no data.** The pipeline runs end to end on synthetic fixtures. Running
it on MIMIC requires PhysioNet credentialed access to MIMIC-IV, MIMIC-IV-ED and
MIMIC-IV-Note, and physician review of the items listed at the end.

## Questions
The Paper 1 claims, tested on real records:
1. Real notes often leave the decision category undetermined.
2. Treating missing inputs as normal under-triages patients.
3. The bounds policy asks only decision-relevant questions and never commits prematurely.
4. A small local model is enough as the extractor.

## Data use (hard constraints, enforced in code)
PhysioNet's guidance on LLMs and credentialed data (2025-09-24,
<https://physionet.org/news/post/llm-responsible-use/>) says the data use agreement forbids
sharing access with third parties, including sending data through APIs. Hence:

| Rule | Where it is enforced |
|---|---|
| Only local models process MIMIC text: `ollama` or `openai_compat` on `localhost`, `127.0.0.1` or `::1`. `anthropic`, `claude_cli` and `session` are refused. | `mimic/config.py` (`assert_local_provider`), at config load and again before a client is created |
| No HTTP(S) proxy may apply to localhost during a run | `assert_no_proxy_for_local`, at the start of `mimic run` |
| No S2 (frontier LLM agent) arm | `MimicPolicy` type: S1, S3, S4, S3-bin only |
| Data, row-level outputs and the LLM cache resolve **outside the repo** (cache entries hold note text; rows hold MIMIC ids) | `MimicPaths.resolved` (symlinks followed) |
| Only aggregate tables (counts, rates, Wilson CIs, means; no ids) leave the output dir | `mimic aggregate`; `aggregate_tables` asserts no id columns |
| Groups smaller than `min_cell_count` (default 10) are suppressed in aggregates | `outcomes.suppress`, `suppress_counts` |

`DataUseError` is not a `ValueError`, so it is never wrapped or caught. `.gitignore` also
ignores common MIMIC file names as a second line of defence.

## Tables used (public column names)
- **MIMIC-IV-ED:** `edstays`, `triage`, `vitalsign`, `diagnosis`.
- **MIMIC-IV hosp:** `patients`, `admissions`, `diagnoses_icd`, `labevents`, `omr`,
  `prescriptions`, `microbiologyevents`.
- **MIMIC-IV icu:** `chartevents` (GCS).
- **MIMIC-IV-Note:** `discharge`.

Each table is read from `<dir>/<table>.csv.gz` (or `.csv`). Large tables are read in chunks,
filtered by item and subject.

## Pipeline
```
mimic cohort      tables -> cohorts -> structured truth -> note sections
                  -> cases.jsonl, notes.jsonl, cohort_flow.csv           (outside the repo)
mimic annotation-template / import-annotations                           (outside the repo)
mimic run         extraction (local model, or the structured oracle) -> bounds
                  -> S1 / S3 / S4 / S3-bin, the EHR as clinician
                  -> extractions/, traces/, rows/<extractor>/*.csv        (outside the repo)
mimic aggregate   rows -> note_alone, policies, questions, extraction, cohort_flow
                  -> <out>/<run_id>/<extractor>/*.csv                    (safe to commit)
```
One case = one (ED stay, calculator). Only ED visits followed by an admission are kept,
because the note is the discharge summary of that admission.

### Cohorts (`mimic/cohorts.py`, definitions in `configs/mimic_criteria.yaml`)
| Calculator | Cohort |
|---|---|
| CURB-65 | ED visits admitted with a pneumonia ICD-9/10 code as principal diagnosis |
| qSOFA | ED visits with suspected infection: a culture then a systemic antibiotic within 72 h, or an antibiotic then a culture within 24 h (Seymour 2016, as in mimic-code), at a time near ED arrival |
| HEART | chest-pain chief complaint (regex on `triage.chiefcomplaint`) or a chest-pain ED diagnosis |
| Cockcroft-Gault | admitted ED visits with creatinine and weight in the structured record |
| PERC / Wells (optional) | D-dimer measured near ED arrival |

`cohort_flow.csv` counts eligible visits, admitted, with a discharge note, selected (optional
seeded sample, `max_cases_per_calculator`), pending annotation, and notes without recognised
headings.

### Structured ground truth (`mimic/truth.py`)
A **partial** assignment: a value missing from the record is Unknown.

| Parameter | Source |
|---|---|
| HR, RR, SBP, DBP, SpO2 | `triage`; else the first `vitalsign` row within `vitals_hours` |
| age, sex | `patients` (age = anchor_age + year of ED arrival − anchor_year) |
| urea | first BUN (mg/dL) within `labs_hours` of arrival, converted to mmol/L by `units.py` |
| creatinine | first creatinine (mg/dL) in the same window |
| HEART troponin | first troponin T, as a level relative to `ref_range_upper` (≤ 1×, 1–3×, > 3×); censored values ("<0.01", "LESS THAN 0.01") only when they settle the level; when `value` is empty the result is read from `comments`, where MIMIC stores some results |
| confusion, altered mentation | first complete GCS in ICU `chartevents`: total < 15 → present, 15 → absent. No GCS (most ED-only patients) → Unknown |
| weight | `omr` "Weight (Lbs)" nearest to arrival (± `weight_days`), converted to kg |
| obesity | `omr` BMI nearest to arrival (> 30); else an obesity code (present) |
| hypertension, hypercholesterolaemia, diabetes, smoking, family history of CAD, atherosclerotic disease, prior VTE, malignancy, haemoptysis | an ICD code of the admission → present. **No code → Unknown, never absent** |
| HEART history, HEART ECG, Wells "PE most likely", DVT signs | judgement items: Unknown until physician annotation |
| other PERC/Wells items | not mapped (Unknown) |

Values outside a parameter's plausible range are dropped (Unknown) and listed per case.

**Reference standard.** The category implied by the structured truth alone, when it is
determined (the bounds over the record collapse to one category). Cases where the record
leaves the category open have no reference and count only in outcomes that need none.

### Physician annotation
Two kinds of item, marked in the template's `analysis` column:
- **primary:** judgement items no structured source can provide (HEART history and ECG,
  Wells "PE most likely", DVT signs). Used in every analysis.
- **secondary:** items the record usually lacks (confusion / altered mentation without an ICU
  GCS). Annotated from the note and used only with `mimic run --secondary` and
  `mimic aggregate --secondary` (outputs under `<extractor>+secondary`). Because this
  reference is read from the same note the model reads, it is reported separately.

`mimic annotation-template CONFIG [--n 100]` writes `annotation_template.csv` (case id,
subject/admission/note ids, item, allowed values, empty `value`). It holds ids only, no note
text, and stays in the output directory. The annotator reads the note in their own MIMIC
environment. `mimic import-annotations CONFIG filled.csv` validates every row (unknown case,
item not pending, invalid value, duplicate) and stores the values with the run. Blank or
`unknown` stays Unknown. Planned: about 100 HEART cases and a sample of Wells cases.

### Note text (`mimic/notes.py`)
MIMIC-IV-Note has no ED provider notes. The extractor reads the discharge summary of the same
admission, restricted to the sections closest to what was known in the ED: chief complaint,
HPI, past medical, social and family history, physical exam and pertinent results. Headings
are matched at line start; a section runs to the next known heading, so "Discharge physical
exam", "Discharge labs" and "Brief hospital course" are cut off. With no recognised heading,
the full text is used and the fallback is counted. Evidence spans index into the selected
text, so the exact-substring check of Paper 1 still applies.

### Extraction, policies, EHR as clinician
- **Extraction:** the Paper 1 prompt, JSON schema and parser
  (`extraction/llm.py`, prompt version recorded with each result), run by a local model,
  cache-first (cache key = request hash; cache in `paths.cache_dir`). Set `num_ctx` on Ollama
  providers so long notes are not truncated. `--extractor oracle` uses the structured truth
  as if the note documented every recorded value: an upper bound for dry runs, not a result.
- **Policies:** S1 ask-all, S3 bounds, S4 bounds + VOI + confidence echo, S3-bin
  (missing = absent), unchanged from Paper 1. S4 on MIMIC uses uncalibrated confidence (no
  labelled dev split) and the synthetic cohort priors for VOI ordering.
- **EHR as clinician** (`mimic/clinician.py`): same interface as the simulated clinician. A
  question is answered with the structured value (including annotations); "not available"
  means the value is truly absent from the record.

## Outcomes (`mimic/outcomes.py`)
MIMIC has no documented-state labels, so extraction is scored against the record **by
extracted state**:

| Extracted | Agrees with the record when |
|---|---|
| present | it scores like the recorded value (same region between the calculator's cuts), equals it for yes/no and graded items, or is within 5% for Cockcroft-Gault inputs; `exact` also requires the same number |
| absent | the recorded value is negative, level 0, or inside the "stated normal" range |
| unknown | no claim; `in_record` gives how often the record had the value |

A disagreement can be an extraction error or a real difference between note and record (for
example, a later measurement). They cannot be told apart without annotation.

| Table | Contents |
|---|---|
| `note_alone.csv` | reference determined; category determined by the note alone; reference within the note's bounds; missing = normal: correct, under- and over-triage |
| `policies.csv` | per policy: accuracy vs reference, abstention, conservative accuracy, under/over-triage, reference within final bounds, premature commitment |
| `questions.csv` | per policy: mean, SD and median questions; "not available" answers; echo questions; irrelevant questions |
| `extraction.csv` | per parameter and extracted state: in record, agrees, exact |
| `cohort_flow.csv` | cohort counts (counts below `min_cell_count` shown as `<N>`) |

Proportions are k/n with Wilson 95% intervals; values not applicable to a case (for example,
no reference) are outside the denominator. Each table has an `all` row pooled over
calculators.

## Running it locally
On the machine that holds the data, with a local Ollama server:
```sh
cp configs/mimic_example.yaml /somewhere/outside/the/repo/mimic.yaml
# edit: paths (all outside the repo), model tags, num_ctx; optionally max_cases_per_calculator
uv run calc-bounds mimic check-config /somewhere/outside/the/repo/mimic.yaml
uv run calc-bounds mimic cohort      /somewhere/outside/the/repo/mimic.yaml
uv run calc-bounds mimic run         /somewhere/outside/the/repo/mimic.yaml               # oracle dry run
uv run calc-bounds mimic run         /somewhere/outside/the/repo/mimic.yaml --extractor qwen9b
uv run calc-bounds mimic annotation-template /somewhere/outside/the/repo/mimic.yaml --n 100
uv run calc-bounds mimic import-annotations  /somewhere/outside/the/repo/mimic.yaml filled.csv
uv run calc-bounds mimic run         /somewhere/outside/the/repo/mimic.yaml --extractor qwen9b  # with annotations
uv run calc-bounds mimic aggregate   /somewhere/outside/the/repo/mimic.yaml --extractor qwen9b --out results/mimic
```
The config file itself can live in the repo (it holds paths, not data), but keeping it outside
avoids committing local paths. Model-size scaling: one extractor entry per local model
(e.g. 4B, 9B, 12B), each run with `--extractor <name>`; the cache makes reruns free.

## Known limitations
- The discharge summary is written after the stay. Even restricted to admission sections, it
  can contain information not available in the ED.
- Only admitted visits are included (discharged ED visits have no discharge summary).
- GCS comes from ICU charting only, so confusion / altered mentation is Unknown for most
  patients unless the note states it.
- Discharge ICD codes undercount comorbidities and can include conditions diagnosed during the
  stay. A missing code is Unknown, so many HEART risk factors stay open and the EHR answers
  "not available".
- The reference standard exists only where the record settles the category; for HEART this
  requires annotation.

## Review items (TODO(physician-review))
All are operational definitions written by us. Code and config carry the marker.

| # | Item | Implemented | Where |
|---|---|---|---|
| M1 | Itemids and omr names | **Verified 2026-10-08** against MIMIC-IV 2.2: GCS 220739 / 223900 / 223901, omr "Weight (Lbs)" and "BMI (kg/m2)", troponin T 51003 (troponin I 51002 / 52642 had no results in the HEART pilot). D-dimer 50915 not yet verified (PERC/Wells not run) | `mimic_criteria.yaml` |
| M2 | Time windows | vitals fallback 4 h; labs [arrival, +24 h]; GCS 24 h; omr ± 365 days; suspicion of infection [−6 h, +24 h] | `mimic_criteria.yaml` |
| M3 | CURB-65 cohort | pneumonia J12–J18 / 480–486 as principal diagnosis | `mimic_criteria.yaml` |
| M4 | qSOFA cohort | Seymour pairing (72 h / 24 h); illustrative antibiotic regex, topical routes excluded; any culture | `mimic_criteria.yaml` |
| M5 | HEART cohort | chief-complaint regex or ED codes R07.1/R07.2/R07.8/R07.9, 786.5 | `mimic_criteria.yaml` |
| M6 | PERC / Wells cohort | D-dimer measured (PERC's low-pretest population is not identifiable) | `mimic_criteria.yaml` |
| M7 | Comorbidity codes | lists per item; history/status codes for atherosclerotic disease; current-use codes for smoking; any C code for malignancy | `mimic_criteria.yaml` |
| M8 | Confusion proxy | CURB-65 confusion = GCS < 15 (same as qSOFA altered mentation); intubated or sedated patients not handled. **Decided (2026-10-08):** GCS exists only for ICU stays, so primary analysis leaves confusion / altered mentation Unknown when there is no GCS; a secondary analysis (`--secondary`) adds them annotated from the note on the annotation sample | `truth.gcs`, `mimic_criteria.yaml` (`secondary_annotation_params`) |
| M9 | Troponin | troponin T vs `ref_range_upper` as the upper reference limit; assay generation and sex-specific cut-offs not modelled (Paper 1 item 27) | `truth.troponin_level` |
| M10 | Triage SpO2 | used as "on room air", which triage does not guarantee (PERC) | `truth.VITALS` |
| M11 | Note sections | sections kept and stop headings; discharge exam and labs cut | `mimic_criteria.yaml` |

Modelling choices (no clinical answer): 5% tolerance for Cockcroft-Gault inputs in extraction
agreement; synthetic priors for S4's VOI ordering; uncalibrated confidence for S4's echo;
first plausible value wins; implausible values dropped using Paper 1's parameter ranges (for
example, urea above 60 mmol/L, BUN about 168 mg/dL, becomes Unknown).
