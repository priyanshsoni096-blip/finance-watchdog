# Baselines on the test splits (development)

Order level. Mean ± 95% Student-t CI over 3 training seeds (LABEL-RULE is deterministic). LABEL-RULE is the definition of the label written as a rule: it is the ceiling, perfect by construction, not a competitor. Size + fast-cancel rule: N = 2, M = 1000 events.

## original test split

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.995 | 0.000 | 0.986 | 1.0 |
| LOGREG | 0.469 ± 0.001 | 0.123 ± 0.000 | 0.528 ± 0.001 | 1.0 |
| SUP-LSTM | 0.995 ± 0.000 | 0.000 ± 0.001 | 0.986 ± 0.000 | 1.0 |
| RL WATCHDOG (cov) | 0.995 ± 0.001 | 0.002 ± 0.002 | 0.984 ± 0.001 | 1.0 |
| Size + fast-cancel rule | 0.941 | 0.812 | 0.352 | — |

## VARIED-SPOOF + ACCUM test split

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.993 | 0.000 | 0.997 | 2.0 |
| LOGREG | 0.303 ± 0.002 | 0.267 ± 0.002 | 0.386 ± 0.002 | 12.0 |
| SUP-LSTM | 0.993 ± 0.000 | 0.000 ± 0.000 | 0.997 ± 0.000 | 2.0 |
| RL WATCHDOG (cov) | 0.994 ± 0.002 | 0.031 ± 0.026 | 0.980 ± 0.014 | 2.0 |
| Size + fast-cancel rule | 0.744 | 0.661 | 0.622 | — |

