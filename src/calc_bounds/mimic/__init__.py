"""Validation on MIMIC-IV (real, credentialed data). Code only: no data lives in this repo.

Pipeline (see docs/MIMIC_VALIDATION.md):
  tables -> cohorts (per calculator) -> structured truth (first ED / admission values)
  -> discharge-note sections -> tri-state extraction (local model) -> bounds
  -> S1 / S3 / S4 / S3-bin with the EHR acting as the clinician -> row-level outputs
  (outside the repo) -> aggregate tables (counts, rates, Wilson CIs; safe to commit).

PhysioNet data use agreement: MIMIC text may only be processed by locally running models.
`config.py` enforces this (local providers only; data, outputs and cache outside the repo).
"""
