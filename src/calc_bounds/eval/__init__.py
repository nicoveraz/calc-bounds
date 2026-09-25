"""Evaluation over traces.

Primary: decision-category accuracy, questions per case (accuracy-vs-questions plot).
Secondary: premature commitment, irrelevant questions, per-param extraction accuracy by
documented state, calibration (Brier, ECE, reliability), silent missing-as-absent errors,
tokens/latency/cost. Paired stats: McNemar (accuracy), Wilcoxon signed-rank (question counts).
Error attribution: routing | extraction | bounds | code.
"""
