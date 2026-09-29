# calc-bounds

[![arXiv](https://img.shields.io/badge/arXiv-2609.34112-b31b1b.svg)](https://arxiv.org/abs/2609.34112)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23004726.svg)](https://doi.org/10.5281/zenodo.23004726)

**Unknown is not normal.** When a clinical note does not mention a finding, that does not mean
the finding is absent. This project tests a safer way to use AI for clinical risk scores
(HEART, CURB-65, qSOFA, PERC, Wells, Cockcroft-Gault):

1. **The AI only reads.** A language model reads the note and marks each score item as
   *present*, *absent* or *not mentioned*, quoting the sentence it relied on.
2. **Ordinary code does the math.** It checks whether the missing items could change the
   decision (for example, low vs moderate risk). If they could not, it gives the answer.
3. **It asks only what matters.** If a missing item could change the decision, it asks the
   clinician about that item and nothing else.

Code for the paper [arXiv:2609.34112](https://arxiv.org/abs/2609.34112) (source: [`paper/manuscript.md`](paper/manuscript.md)).

## Main results in plain language
Tested on 1,200 synthetic emergency patients, with a simulated clinician answering questions.

- **Assuming "not mentioned = normal" puts about 1 in 12 patients in a lower-risk group than
  they belong to (8.5%).** This is how many tools and benchmarks treat missing data. The risk
  stayed the same with messy notes and with a clinician who sometimes misremembers (8–10%).
  Our approach did this in about 1 in 1,000 patients (0.1%).
- **About one question per patient instead of two, with the same accuracy.** Asking for every
  missing item took 1.8 questions per patient, and about half of those questions could not
  have changed the decision. Our approach asked 0.9 questions per patient, every one of them
  capable of changing the result. Both got the risk group right in 99.4% of patients.
- **It never answers too early.** It gives a risk group only when the answer can no longer
  change.
- **A leading AI agent working alone (Claude Opus 5.5) did well in ideal conditions but less
  well in realistic ones.**
  - When the clinician always answered correctly, it was as accurate (99.6%), but about 1 in
    10 of its questions could not have changed the decision.
  - When the clinician sometimes did not know, misremembered, or answered vaguely, it was
    less accurate than our approach (83.5% vs 87.0%). It also gave an answer before it had
    enough information in about 3% of patients.
- **A small AI model that runs on a laptop was enough for the reading step.** Accuracy
  matched the perfect-reading baseline (99.8%). Patient text never has to leave the
  computer.
- **Real notes are often incomplete.** In 584 published case reports (MedCalc-Bench), only
  about half (52%) had enough information to settle the risk group. For HEART it was 13%.
  There, "not mentioned = normal" gave the right risk group only 69% of the time.

**Bottom line for clinicians:** an AI that fills gaps by assuming "normal" will under-rate
some patients' risk without warning. Letting the AI only read the note, and letting code
decide what is still unknown and whether it matters, avoids this. You are asked only the
questions that could change the decision.

**Limits:** the patients and notes are synthetic, the clinician is simulated, and the
calculator readings were reviewed by one physician. Validation on real hospital records is
future work.

Details: [`docs/RESULTS.md`](docs/RESULTS.md) · tables and figures: [`paper/`](paper/)

## Scope
- **Calculators:** HEART, CURB-65, qSOFA, PERC, Wells (PE, two-tier) and Cockcroft-Gault.
  Each cites its primary source; open clinical questions are listed in
  [`docs/CALCULATOR_NOTES.md`](docs/CALCULATOR_NOTES.md).
- **Policies** (config id in brackets):

  | Policy | What it does |
  |--------|--------------|
  | Ask-all (`s1_ask_all`) | asks for every missing input |
  | Agent (`s2_llm_agent`) | end-to-end LLM agent with ask / calculate / answer actions |
  | Bounds (`s3_bounds`) | tri-state extraction + bounds; asks only decision-relevant inputs |
  | Bounds + checks (`s4_bounds_voi_echo`) | Bounds + value-of-information ordering + confirmation of low-confidence values |
  | Missing = normal (`s3_bin`) | Bounds with missing inputs treated as normal (the common convention) |

- **Conditions:** clean or messy notes × ideal or noisy simulated clinician.

## Reproduce
```sh
uv sync
uv run pytest                      # 150 tests, including hypothesis property tests of the bounds
scripts/reproduce_paper.sh         # regenerates every table and figure
```
Only the code is released. The synthetic notes, extractions, traces and the LLM response cache
(`.cache/llm`) are not distributed. Running the script therefore re-queries the models, at $0:
- **Claude models** run through headless Claude Code (`claude -p`, subscription): notes by
  Sonnet 5, judge and agent Opus 5.5, extractor Haiku 4.5.
- **Qwen3.5-9B** runs locally through Ollama (`ollama pull qwen3.5:9b`).

The cohort is regenerated exactly from the seed. LLM outputs (notes, extractions, agent turns)
are not guaranteed to be byte-identical, so re-run numbers can differ slightly from the paper.
With a populated cache, the same commands make no model calls.

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
Paper: [arXiv:2609.34112](https://arxiv.org/abs/2609.34112). Code: [MIT](LICENSE). Archived at Zenodo: [10.5281/zenodo.23004726](https://doi.org/10.5281/zenodo.23004726). Citation: see [`CITATION.cff`](CITATION.cff). MedCalc-Bench data is not included (CC-BY-SA 4.0; see `docs/ANCHOR.md`).
