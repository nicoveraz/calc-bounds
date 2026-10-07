# JAMIA submission checklist

Article type: Research and Applications. Check the author instructions again at submission:
<https://academic.oup.com/jamia/pages/General_Instructions>.

## Before running Study 2
- [ ] MIMIC-IV-Note 2.2 downloaded and checksums verified, outside any repository.
- [ ] Physician review of operational definitions M1–M11 (`docs/MIMIC_VALIDATION.md`).
- [ ] Item ids checked against `d_labitems` / `d_items` (M1).

## Study 2 runs (local only)
- [ ] `calc-bounds mimic check-config`, `cohort`, oracle dry run.
- [ ] HEART judgement items annotated (~100 cases), imported.
- [ ] Runs with 4B, 9B, 12B local extractors; `mimic aggregate` to `results/mimic/`.
- [ ] Figure 5 and Tables 2–3 generated from the aggregate tables (never hand-computed).

## Manuscript
- [ ] `python3 paper/jamia/check.py` reports 0 placeholders; abstract <= 250 words, main
      text <= 4,000, <= 4 tables, <= 6 figures.
- [ ] Supplement: add the MIMIC operational definitions section (S-MIMIC).
- [ ] Second physician reviews calculator readings and MIMIC definitions (co-author?).
- [x] MIMIC reference DOIs checked against the PhysioNet project pages (2026-10-07).
- [ ] Confirm the ethics statement (deidentified data under the PhysioNet DUA).
- [ ] AI-use disclosure in Methods and cover letter (done in the draft).
- [ ] Data availability statement (done in the draft).

## Release and submission
- [ ] arXiv v2 with the MIMIC results, before submission (JAMIA allows preprints; after
      acceptance the preprint may not be replaced by the published version, only linked).
- [ ] `paper/jamia/build.sh` -> `manuscript.docx`; figures uploaded as separate files.
- [ ] Choose standard (subscription) publication: no charge. Open access is optional.
- [ ] After acceptance: update the arXiv entry with the journal DOI.
