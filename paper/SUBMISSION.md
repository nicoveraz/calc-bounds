# arXiv submission metadata

Everything the arXiv form asks for, in the order it asks, in the same format as the author's
earlier arXiv papers. Nothing here is uploaded automatically: `./build.sh` builds and verifies
`arxiv-submission.tar.gz`; the fields below are pasted by hand.

## Before uploading
1. Make https://github.com/nicoveraz/calc-bounds public (the paper links to it).
2. Run `./build.sh`. It must end with `OK -- verified from the tarball's own contents.`
3. Upload `arxiv-submission.tar.gz` (main.tex, main.bbl, refs.bib, figures/). Check arXiv's
   compiled PDF against `arxiv.pdf` before confirming.

## Title
```
Unknown is not normal: separating language-model extraction from rule-based decision logic for clinical risk scores
```
No dashes in the title field (arXiv renders them as two literal hyphens).

## Authors
```
Nicolás Vera Zúñiga
```
Independent Researcher, Chile. `nicovera@quetru.cl`.

## Categories
- Primary: `cs.CL` (same as the author's earlier papers).
- Cross-list: `cs.AI`.

## License
```
CC BY 4.0
```

## Comments
```
14 pages (7 of main text), 5 figures, 2 tables, appendix included; full supplementary material in the code repository. Code: https://github.com/nicoveraz/calc-bounds (archived: https://doi.org/10.5281/zenodo.23004726)
```
arXiv does not allow editing Comments after announcement without a new version, so check it now.

## Abstract (plain text, ready to paste)
arXiv caps this field at 1,920 characters. This version is 1733 characters.
```
Large language models (LLMs) are increasingly used to compute clinical risk scores from free-text notes. Notes are often incomplete, and treating undocumented findings as normal can silently misclassify patients. We test whether separating three-state extraction (present, absent or unknown, by an LLM) from decision logic (deterministic code computing score bounds over unknown inputs) lets a system ask only questions that can change the decision. On 1,200 synthetic emergency cases across six calculators (HEART, CURB-65, qSOFA, PERC, Wells, Cockcroft-Gault), with a simulated clinician answering questions, we compared this bounds policy with asking for every missing input, a missing-equals-normal schema, and an end-to-end LLM agent (Claude Opus 5.5). With Claude Haiku 4.5 as extractor, the bounds policy matched ask-all accuracy (99.4% vs 99.4%) with half the questions (0.92 vs 1.78 per case) and no irrelevant ones. Treating missing as normal dropped accuracy to 91.2% and under-triaged 8.5% of patients (95% CI 7.1-10.2), and under-triage persisted under messy notes and a noisy clinician. The agent was equally accurate under ideal conditions (99.6%) but 9.5% of its questions were irrelevant; with a noisy clinician it was less accurate than the bounds policy (83.5% vs 87.0%, p<0.001) and committed prematurely in 2.7% of cases (bounds: 0%). A 9B local model as extractor reached oracle-level accuracy (99.8%). In 584 real case reports from MedCalc-Bench, only 52% contained enough information to determine the category (HEART 13%). Routing decisions through code that reasons explicitly about unknowns avoids premature commitment and irrelevant questions, halves the questions asked, and works with small local models.
```
