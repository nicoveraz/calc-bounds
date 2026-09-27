# calc-bounds

**Unknown is not normal.** A language model reads a clinical note into typed facts, each marked
*present*, *absent* or *unknown* with an exact evidence span. Deterministic code then computes a
clinical risk score's possible values over the unknowns (its *bounds*). It decides whether the
decision category is already determined and, if not, asks the clinician only for the facts
that could change it.

Code and data for Paper 1 (draft: [`paper/manuscript.md`](paper/manuscript.md)).

## Main results (English)
- **Treating undocumented findings as normal under-triages about 1 in 12 patients.** This
  held in every condition tested (8.2–10.2%), even when headline accuracy looked fine. The
  bounds policy under-triaged 0.0–0.5%.
- **The bounds policy asks half as many questions as asking for everything, at the same
  accuracy.** It asks no irrelevant questions and never commits before the category is
  determined.
- **A frontier end-to-end agent (Opus 5.5) is competitive on accuracy** under ideal
  conditions. It is worse under a noisy clinician, commits prematurely in up to 3.4% of
  cases, and about 10% of its questions are irrelevant.
- **A 9B local model (Qwen3.5-9B) as extractor reaches oracle-level accuracy.**
- **On 584 real case reports (MedCalc-Bench Verified), only 52% determine the category** from
  the note alone. HEART: 13%.

Details: [`docs/RESULTS.md`](docs/RESULTS.md) · tables and figures: [`paper/`](paper/)

## Scope
- **Calculators:** HEART, CURB-65, qSOFA, PERC, Wells (PE, two-tier) and Cockcroft-Gault.
  Each cites its primary source; open clinical questions are listed in
  [`docs/CALCULATOR_NOTES.md`](docs/CALCULATOR_NOTES.md).
- **Systems:**

  | id | system |
  |----|--------|
  | S1 | ask for every missing input |
  | S2 | end-to-end LLM agent with ask / calculate / answer actions |
  | S3 | tri-state extraction + bounds; ask only decision-relevant inputs |
  | S4 | S3 + value-of-information ordering + confirmation of low-confidence values |
  | S3-bin | S3 with missing inputs treated as normal (the common convention) |

- **Conditions:** clean or messy notes × ideal or noisy simulated clinician.

## Reproduce
```sh
uv sync
uv run pytest                      # 149 tests, including hypothesis property tests of the bounds
scripts/reproduce_paper.sh         # regenerates every table and figure
```
With the LLM cache (`.cache/llm`) present, no model calls are made. Without it, the same
commands re-run the models at $0:
- **Claude models** run through headless Claude Code (`claude -p`, subscription): notes by
  Sonnet 5, judge and agent Opus 5.5, extractor Haiku 4.5.
- **Qwen3.5-9B** runs locally through Ollama (`ollama pull qwen3.5:9b`).

Every run is one YAML config ([`configs/main.yaml`](configs/main.yaml)) with a fixed seed.
Every model call is cached by a hash of provider, model, prompt and parameters.

The real-note anchor needs MedCalc-Bench Verified in `data/raw/medcalc/` (CC-BY-SA 4.0; not
redistributed). See [`docs/ANCHOR.md`](docs/ANCHOR.md).

## Data policy
- The cohort and notes are **synthetic**.
- The only real text is the published, de-identified MedCalc-Bench case reports used for the
  anchor. They are kept in the gitignored `data/raw/`.

## Layout
```
src/calc_bounds/   calculators/ bounds/ cohort/ render/ extraction/ llm/ simulator/
                   policies/ eval/ anchor/ pipeline.py cli.py
configs/           run configs (main.yaml = the paper)
docs/              ARCHITECTURE, RESULTS, CALCULATOR_NOTES, ANCHOR, RENDER_LOG, RELATED_WORK
paper/             manuscript.md, tables/, figures/
scripts/           reproduce_paper.sh
```

## License and citation
Code: [MIT](LICENSE). Citation: see [`CITATION.cff`](CITATION.cff). MedCalc-Bench data is not included (CC-BY-SA 4.0; see `docs/ANCHOR.md`).
