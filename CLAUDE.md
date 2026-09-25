# CLAUDE.md — calc-bounds

Research repo. Full spec: `SPEC.md`. Architecture/interfaces: `docs/ARCHITECTURE.md`.
Idea: model extracts tri-state typed params with calibrated confidence; deterministic code
computes the score and decides which missing params are decision-relevant (score bounds).

## Hard rules
- **Synthetic data only.** Never add real patient data. `data/raw/`, `data/private/`, `.env`,
  caches and `runs/` are gitignored.
- **Unknown is never Absent** in scoring/bounds code. Only exception: the S3-bin ablation.
- **No clinical criteria from memory.** Each calculator module docstring cites its primary
  source. Uncertain criterion/threshold/points/unit → implement best reading, mark
  `# TODO(physician-review)`, and list it in `docs/CALCULATOR_NOTES.md`.
- **Units are normalized by code** (`units.py`), never by the model.
- **Evidence spans** must be exact substrings of the note (`note[start:end] == text`);
  reject and log mismatches.
- **Verify package versions and SDK APIs** before use; never hardcode model names or prices
  (they live in config).
- **Dependencies:** only pydantic v2, pytest, hypothesis, pandas, numpy, scipy, statsmodels,
  matplotlib, pyyaml, typer, anthropic, openai (local OpenAI-compatible servers only),
  scikit-learn (calibration only), ruff. Ask the user before adding anything else.
- **No frameworks:** no LangChain/LlamaIndex/agent frameworks, no DB, no web UI. JSONL + pandas.
- **Reproducibility:** every run = one YAML config, fully seeded; all LLM calls through the disk
  cache keyed by sha256(provider, model, messages, tools, params); track tokens and cost.
- **Stop at the end of each milestone** (see SPEC.md) with a summary; wait for approval.
- Ask the user when a clinical or technical decision is genuinely ambiguous.

## Conventions
- Python 3.12, `uv`, src layout (`src/calc_bounds`). Type hints everywhere; pydantic models at
  module boundaries. Readable over clever; no plugin systems, factories, deep hierarchies.
- Small focused commits with clear messages.
- Tests: `tests/`, hypothesis property tests for bounds are mandatory. Tests that hit a real
  LLM are marked `@pytest.mark.llm` and are skipped by default.

## Commands
```sh
uv sync                                  # install
uv run pytest                            # tests (excludes llm-marked)
uv run pytest -m llm                     # real-API tests only
uv run ruff check . && uv run ruff format .
uv run calc-bounds --help
uv run calc-bounds check-config configs/m2_oracle.example.yaml
```
