| Condition | Comparison | Accuracy A vs B, % | Discordant (A only / B only) | McNemar p (Holm) | Questions A vs B | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|
| Oracle, clean, ideal | Bounds vs Ask-all | 99.8 vs 99.8 | 0 / 0 | 1.000 | 0.92 vs 1.74 | <0.001 |
| Oracle, clean, ideal | Bounds vs Agent | 99.8 vs 99.6 | 3 / 0 | 0.750 | 0.92 vs 0.99 | <0.001 |
| Oracle, clean, ideal | Bounds vs Missing = normal | 99.8 vs 91.8 | 99 / 2 | <0.001 | 0.92 vs 0.21 | <0.001 |
| Oracle, clean, ideal | Bounds + checks vs Bounds | 99.8 vs 99.8 | 0 / 0 | 1.000 | 0.87 vs 0.92 | <0.001 |
| Haiku, clean, ideal | Bounds vs Ask-all | 99.4 vs 99.4 | 0 / 0 | 1.000 | 0.92 vs 1.78 | <0.001 |
| Haiku, clean, ideal | Bounds vs Agent | 99.4 vs 99.6 | 3 / 5 | 1.000 | 0.92 vs 0.99 | <0.001 |
| Haiku, clean, ideal | Bounds vs Missing = normal | 99.4 vs 91.2 | 101 / 2 | <0.001 | 0.92 vs 0.21 | <0.001 |
| Haiku, clean, ideal | Bounds + checks vs Bounds | 99.7 vs 99.4 | 3 / 0 | 0.750 | 0.90 vs 0.92 | 0.005 |
| Qwen, clean, ideal | Bounds vs Ask-all | 99.8 vs 99.8 | 0 / 0 | 1.000 | 1.23 vs 2.21 | <0.001 |
| Qwen, clean, ideal | Bounds vs Agent | 99.8 vs 99.6 | 3 / 0 | 0.750 | 1.23 vs 0.99 | <0.001 |
| Qwen, clean, ideal | Bounds vs Missing = normal | 99.8 vs 91.2 | 105 / 2 | <0.001 | 1.23 vs 0.26 | <0.001 |
| Qwen, clean, ideal | Bounds + checks vs Bounds | 99.8 vs 99.8 | 0 / 0 | 1.000 | 1.20 vs 1.23 | 0.004 |
| Haiku, clean, noisy | Bounds vs Ask-all | 87.0 vs 87.0 | 0 / 0 | 1.000 | 0.96 vs 1.78 | <0.001 |
| Haiku, clean, noisy | Bounds vs Agent | 87.0 vs 83.5 | 65 / 23 | <0.001 | 0.96 vs 1.03 | <0.001 |
| Haiku, clean, noisy | Bounds vs Missing = normal | 87.0 vs 88.0 | 70 / 82 | 0.750 | 0.96 vs 0.21 | <0.001 |
| Haiku, clean, noisy | Bounds + checks vs Bounds | 87.2 vs 87.0 | 3 / 0 | 0.750 | 0.94 vs 0.96 | 0.078 |
