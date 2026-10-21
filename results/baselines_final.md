# Baselines on every held-out dataset (scored once, after the baselines were committed)

Order level. Mean ± 95% Student-t CI over 3 training seeds (LABEL-RULE is deterministic). LABEL-RULE is the definition of the label written as a rule: it is the ceiling, perfect by construction, not a competitor. Size + fast-cancel rule: N = 2, M = 1000 events.

## original test split

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.995 | 0.000 | 0.986 | 1.0 |
| LOGREG | 0.469 ± 0.001 | 0.123 ± 0.000 | 0.528 ± 0.001 | 1.0 |
| SUP-LSTM | 0.995 ± 0.000 | 0.000 ± 0.001 | 0.986 ± 0.000 | 1.0 |
| RL WATCHDOG (cov) | 0.995 ± 0.001 | 0.002 ± 0.002 | 0.984 ± 0.001 | 1.0 |
| Size + fast-cancel rule | 0.941 | 0.812 | 0.352 | — |

## held-out split (SCRIPTED-ATK, AMZN)

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.990 | 0.000 | 0.995 | 7.0 |
| LOGREG | 0.198 ± 0.001 | 0.000 ± 0.000 | 0.329 ± 0.001 | 8.0 |
| SUP-LSTM | 0.990 ± 0.000 | 0.000 ± 0.000 | 0.995 ± 0.000 | 7.0 |
| RL WATCHDOG (cov) | 0.990 ± 0.002 | 0.004 ± 0.012 | 0.994 ± 0.002 | 7.0 |
| Size + fast-cancel rule | 0.940 | 0.869 | 0.604 | — |

## LATEBURST-ATK

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.983 | 0.000 | 0.992 | 16.0 |
| LOGREG | 0.218 ± 0.004 | 0.098 ± 0.000 | 0.300 ± 0.005 | 13.3 |
| SUP-LSTM | 0.983 ± 0.000 | 0.000 ± 0.001 | 0.991 ± 0.001 | 16.0 |
| RL WATCHDOG (cov) | 0.984 ± 0.001 | 0.002 ± 0.002 | 0.987 ± 0.003 | 16.0 |
| Size + fast-cancel rule | 0.885 | 0.819 | 0.462 | — |

## DRIP-ATK

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.989 | 0.000 | 0.994 | 7.0 |
| LOGREG | 0.324 ± 0.001 | 0.144 ± 0.001 | 0.358 ± 0.001 | 10.0 |
| SUP-LSTM | 0.989 ± 0.000 | 0.000 ± 0.000 | 0.994 ± 0.001 | 7.0 |
| RL WATCHDOG (cov) | 0.989 ± 0.002 | 0.012 ± 0.015 | 0.973 ± 0.025 | 7.0 |
| Size + fast-cancel rule | 0.744 | 0.797 | 0.336 | — |

## REVERSAL-ATK

| Detector | Recall | Legitimate FPR | Pooled F1 | Median delay (steps) |
|---|---|---|---|---|
| LABEL-RULE | 0.984 | 0.000 | 0.992 | 7.0 |
| LOGREG | 0.218 ± 0.003 | 0.099 ± 0.000 | 0.314 ± 0.003 | 5.0 |
| SUP-LSTM | 0.984 ± 0.000 | 0.001 ± 0.001 | 0.991 ± 0.001 | 7.0 |
| RL WATCHDOG (cov) | 0.983 ± 0.005 | 0.012 ± 0.015 | 0.981 ± 0.010 | 7.0 |
| Size + fast-cancel rule | 0.964 | 0.804 | 0.591 | — |

