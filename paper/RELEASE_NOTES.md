# v1.0.1: arXiv:2609.34112

Metadata only; no code or results changed since v1.0.0.
- The paper is published as [arXiv:2609.34112](https://arxiv.org/abs/2609.34112) (cs.CL, cross-listed cs.AI).
- README, `CITATION.cff` (preferred citation) and `.zenodo.json` now name the paper, so the
  Zenodo record links to it.
- The Zenodo concept DOI (10.5281/zenodo.23004726) is added to the paper source, README and
  citation file; the v1.0.0 snapshot predates it.

# v1.0.0: Paper 1 (arXiv submission)

Code for *Unknown is not normal: separating language-model extraction from rule-based
decision logic for clinical risk scores* (Vera Zúñiga, 2026).

## What is included
- Six calculators with primary-source citations and score bounds over unknown inputs
  (property-tested with Hypothesis).
- Seeded synthetic cohort generator, note rendering and validation, extraction
  (Claude via headless Claude Code, Qwen3.5-9B via Ollama), five policies (ask-all, agent,
  bounds, bounds + checks, missing = normal), a simulated clinician (ideal and noisy), and
  evaluation.
- `scripts/reproduce_paper.sh` and `calc-bounds paper` regenerate every table and figure;
  `paper/build.sh` builds the PDFs and the verified arXiv source package.
- The paper as submitted: `paper/arxiv.pdf`, with the full supplementary material in
  `paper/supplement.pdf`.

## Not included
Synthetic notes, extractions, traces and the model-response cache, and MedCalc-Bench Verified
(CC-BY-SA 4.0; obtain it from its authors). Re-running re-queries the models, so outputs can
differ slightly from the paper.

Code: MIT licence.
