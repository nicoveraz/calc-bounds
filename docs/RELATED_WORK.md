# Related work and benchmarks (verified 2026-09-26)

No existing benchmark tests the whole task: computing a clinical score from an incomplete
note by asking a clinician.

**Closest prior work: ClinDet-Bench.** Watanabe et al., arXiv 2602.22771 (2026),
github.com/yusukewatanabe1208/ClinDet_Benchmark, MIT.
- Cases can be:
  - complete;
  - incomplete but determinable (the missing items can't cross the threshold);
  - incomplete and undeterminable.

  This is the same min/max-against-threshold logic as our bounds.
- 94 synthetic scenarios and 16 scores, including CURB-65 and qSOFA.
- There is no asking step: models judge or abstain only. It finds LLMs both commit too early
  and over-abstain.
- **Our study is essentially ClinDet-Bench plus question asking.** It can serve as an
  external check for CURB-65 and qSOFA.

**Interaction harnesses (diagnosis, not scores):**
- **MediQ** (Li et al., NeurIPS 2024; arXiv 2406.00922; CC-BY-4.0). An expert model asks a
  simulated patient questions, scored on accuracy against number of questions (our metric).
- **AgentClinic** (Schmidgall et al., arXiv 2405.07960; MIT).
- **CRAFT-MD** (Johri et al., *Nature Medicine* 2025, doi:10.1038/s41591-024-03328-5).

**Calculator benchmarks:**
- **MedCalc-Bench** (Khandekar et al., NeurIPS 2024 Datasets & Benchmarks). The Verified
  release is our anchor (see ANCHOR.md).
- **MedRaC / "From Scores to Steps"** (EMNLP 2025): step-level re-evaluation of
  MedCalc-Bench, with no missing-information component.
- **MedMCP-Calc** (ACL 2026; arXiv 2601.23049): agents query a MIMIC-derived EHR database
  for missing inputs. Heavy, and probably needs credentialed access.
- **MedCalc-Pro** (arXiv 2607.02879): about choosing the right calculator, not about missing
  information.

**Label quality:**
- Ye et al., arXiv 2512.19691: relabeling of MedCalc-Bench v1.0 with physician oversight.
- Krohn-Grimberghe, arXiv 2603.02222: calculator implementation bugs in MedCalc-Bench.

**Omissions:**
- "LLM Judges Verify Presence, Not Absence" (arXiv 2608.31016): LLM judges miss omissions in
  generated notes. This bears on our judge-based note validation.
