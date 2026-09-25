# calc-bounds

Experiment: does separating **calibrated tri-state extraction** (an LLM) from
**decision-relevance computation** (deterministic code, via score bounds) reduce the questions
asked of a clinician without losing decision-category accuracy, compared with an end-to-end LLM
agent?

A model extracts each calculator parameter from a synthetic clinical note as
`Present(value, unit, confidence, evidence)`, `Absent(confidence, evidence)` or `Unknown()`.
Code computes the score and the set of decision categories still possible, and only asks about a
missing parameter when its value could change the category.

**Synthetic data only.** No real patient data belongs in this repository.

## Systems compared
| id | description |
|----|-------------|
| S1 | extract, then ask for every missing parameter |
| S2 | end-to-end LLM agent with `calculate` and `ask_clinician` tools |
| S3 | tri-state extraction + bounds: ask only decision-relevant params |
| S4 | S3 + value-of-information ordering + confidence echo |
| S3-bin | S3 ablation with a binary schema (missing = absent) |

Calculators: HEART, CURB-65, qSOFA, PERC, Wells (PE, two-tier), Cockcroft-Gault.

## Setup
```sh
uv sync
uv run pytest
uv run calc-bounds --help
```
API keys go in environment variables (named in the run config), never in files under git.

## Layout
See `docs/ARCHITECTURE.md`. Status: Milestone 0 (scaffold only; no logic yet).
