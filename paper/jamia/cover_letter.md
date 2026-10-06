# Cover letter (draft)

Dear Editor,

Please consider the enclosed manuscript, "Unknown is not normal: separating language-model
extraction from rule-based decision logic for clinical risk scores", for publication as a
Research and Applications article in JAMIA.

Large language models are increasingly used to compute clinical risk scores from notes, and
the standard benchmark treats any input a note does not mention as absent or normal. We show
that this convention is a patient-safety problem: it under-triaged about one patient in twelve
across 1,200 synthetic emergency cases, gave the right category for only two-thirds of
published case reports, and [[MIMIC: result on real emergency visits in MIMIC-IV]].

We propose and evaluate a simple alternative that fits existing clinical decision support: a
language model only reads the note into present, absent or unknown facts, and deterministic
code decides whether the documented facts determine the category, asking the clinician only
what can change it. It matched the accuracy of asking for every missing input with half the
questions, never committed prematurely, never asked an irrelevant question, and works with a
small model running locally, which matters when records cannot leave the institution.

The work extends a preprint (arXiv:2609.34112) with a validation on MIMIC-IV records, run
entirely with local models in accordance with the PhysioNet data use agreement. Code is open
(MIT) and archived on Zenodo.

Use of AI: language models generated the synthetic notes and served as extractors, judge and
agent in the experiments, as described in the Methods; code and analysis were developed with
AI assistance (Claude Code). No MIMIC data was processed by any remote model.

The manuscript has not been published elsewhere and is not under consideration by another
journal. The author has no conflicts of interest and received no funding for this work.

Sincerely,

Nicolás Vera Zúñiga, MD
Independent researcher, Chile
nicovera@quetru.cl
