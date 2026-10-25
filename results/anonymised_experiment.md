# Detection without participant attribution

Every other detector in this project is handed the monitored participant's own trade direction and the side of its resting orders — together, the manipulation label's definition. This experiment removes that attribution entirely and uses **only the displayed order book**, as an external surveillance system on an anonymised feed would see it.

Data: `C:\Users\Rahul Soni\Desktop\finance-watchdog\data\watchdog_v2`. Splits are disjoint stretches of the trading day; preprocessing and every threshold are fitted on training data only; episodes never cross splits. **3 seeds** (0, 1, 2), reported as mean ± 95% Student-t CI.

## Leakage audit of the retained features

| Check | Value |
|---|---|
| Steps in train | 360,000 |
| Positive rate (causal label) | 0.111 |
| **Withheld pair** (features 44 + 45) step F1 — what we removed | **0.948** |
| Largest absolute correlation of any retained feature | 0.381 (`book L1 bid size`) |
| Retained features with correlation > 0.5 | 0 |
| Best single retained feature rule (swept on train) | 0.490 (`total displayed depth`) |

The withheld pair reconstructs the label almost perfectly; no retained feature comes close. The removal is real — the label is not reachable by another column. Full definitions in `docs/FEATURE_DICTIONARY.md`.

## Step level — is manipulation happening now?

| Model | Split | Precision | Recall | F1 | FPR | PR-AUC |
|---|---|---|---|---|---|---|
| constant: never flag | test (in-distribution) | n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.061 ± 0.000 |
| constant: never flag | held-out | n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.096 ± 0.000 |
| constant: always flag | test (in-distribution) | 0.116 ± 0.000 | 1.000 ± 0.000 | 0.207 ± 0.000 | 1.000 ± 0.000 | 0.061 ± 0.000 |
| constant: always flag | held-out | 0.122 ± 0.000 | 1.000 ± 0.000 | 0.218 ± 0.000 | 1.000 ± 0.000 | 0.096 ± 0.000 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | test (in-distribution) | 0.046 ± 0.000 | 0.029 ± 0.000 | 0.036 ± 0.000 | 0.079 ± 0.000 | 0.062 ± 0.000 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | held-out | 0.120 ± 0.000 | 0.085 ± 0.000 | 0.099 ± 0.000 | 0.086 ± 0.000 | 0.100 ± 0.000 |
| logistic regression | test (in-distribution) | 0.491 ± 0.000 | 0.851 ± 0.000 | 0.622 ± 0.000 | 0.115 ± 0.000 | 0.500 ± 0.000 |
| logistic regression | held-out | 0.409 ± 0.000 | 0.376 ± 0.000 | 0.392 ± 0.000 | 0.075 ± 0.000 | 0.314 ± 0.000 |
| gradient boosting | test (in-distribution) | 0.604 ± 0.014 | 0.584 ± 0.026 | 0.594 ± 0.007 | 0.050 ± 0.005 | 0.627 ± 0.010 |
| gradient boosting | held-out | 0.540 ± 0.016 | 0.125 ± 0.015 | 0.203 ± 0.020 | 0.015 ± 0.002 | 0.321 ± 0.002 |
| supervised LSTM | test (in-distribution) | 0.878 ± 0.023 | 0.850 ± 0.042 | 0.864 ± 0.020 | 0.015 ± 0.004 | 0.933 ± 0.026 |
| supervised LSTM | held-out | 0.311 ± 0.023 | 0.028 ± 0.007 | 0.052 ± 0.011 | 0.009 ± 0.002 | 0.166 ± 0.015 |

## Episode level — does this window contain manipulation?

| Model | Split | Precision | Recall | F1 | FPR | PR-AUC |
|---|---|---|---|---|---|---|
| constant: never flag | test (in-distribution) | n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — |
| constant: never flag | held-out | n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — |
| constant: always flag | test (in-distribution) | 0.257 ± 0.000 | 1.000 ± 0.000 | 0.408 ± 0.000 | 1.000 ± 0.000 | — |
| constant: always flag | held-out | 0.625 ± 0.000 | 1.000 ± 0.000 | 0.769 ± 0.000 | 1.000 ± 0.000 | — |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | test (in-distribution) | 0.183 ± 0.000 | 0.357 ± 0.000 | 0.242 ± 0.000 | 0.552 ± 0.000 | — |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | held-out | 0.515 ± 0.000 | 0.624 ± 0.000 | 0.564 ± 0.000 | 0.980 ± 0.000 | — |
| logistic regression | test (in-distribution) | 0.380 ± 0.000 | 0.974 ± 0.000 | 0.546 ± 0.000 | 0.549 ± 0.000 | — |
| logistic regression | held-out | 0.685 ± 0.000 | 0.836 ± 0.000 | 0.753 ± 0.000 | 0.640 ± 0.000 | — |
| gradient boosting | test (in-distribution) | 0.534 ± 0.014 | 0.892 ± 0.025 | 0.668 ± 0.007 | 0.268 ± 0.021 | — |
| gradient boosting | held-out | 0.975 ± 0.013 | 0.416 ± 0.010 | 0.583 ± 0.010 | 0.018 ± 0.010 | — |
| supervised LSTM | test (in-distribution) | 0.709 ± 0.046 | 0.968 ± 0.016 | 0.818 ± 0.027 | 0.138 ± 0.032 | — |
| supervised LSTM | held-out | 0.732 ± 0.048 | 0.288 ± 0.034 | 0.413 ± 0.043 | 0.176 ± 0.025 | — |

## Confusion matrices — held-out, step level, seed 0

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| constant: never flag | 0 | 0 | 9,781 | 70,219 |
| constant: always flag | 9,781 | 70,219 | 0 | 0 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | 828 | 6,066 | 8,953 | 64,153 |
| logistic regression | 3,673 | 5,300 | 6,108 | 64,919 |
| gradient boosting | 1,219 | 1,068 | 8,562 | 69,151 |
| supervised LSTM | 252 | 583 | 9,529 | 69,636 |

## Calibration (Brier score, lower is better)

| Model | test (in-distribution) | held-out |
|---|---|---|
| constant: never flag | 0.116 | 0.122 |
| constant: always flag | 0.884 | 0.878 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | 0.182 | 0.188 |
| logistic regression | 0.110 | 0.139 |
| gradient boosting | 0.080 | 0.110 |
| supervised LSTM | 0.029 | 0.128 |

A classifier that always predicted the base rate (0.111) would score 0.099. A model must beat that to be adding probabilistic information, not just ranking information.

## Per source — held-out, step level, seed 0

Instrument and attack family are both visible here: `SCRIPTED-ATK_*` is the unseen scripted attacker, `SPOOFER-05_AMZN` an unseen RL agent on an unseen stock, and `HONEST_*` / `FLICKER_*` are legitimate controls (no positives, so F1 is undefined and the FPR is what matters).

| Source | Positives / steps | constant: never flag F1 | constant: always flag F1 | book-imbalance rule (|deep imbalance| > p90 = 0.279) F1 | logistic regression F1 | gradient boosting F1 | supervised LSTM F1 |
|---|---|---|---|---|---|---|---|
| FLICKER_AMZN | 0 / 10,000 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HONEST_AMZN | 0 / 10,000 | n/a | 0.000 | 0.000 | 0.000 | n/a | 0.000 |
| SCRIPTED-ATK_AAPL | 1,764 / 10,000 | 0.000 | 0.300 | 0.176 | 0.073 | 0.000 | 0.004 |
| SCRIPTED-ATK_AMZN | 2,001 / 10,000 | 0.000 | 0.333 | 0.170 | 0.131 | 0.006 | 0.022 |
| SCRIPTED-ATK_GOOG | 1,762 / 10,000 | 0.000 | 0.300 | 0.166 | 0.126 | 0.011 | 0.007 |
| SCRIPTED-ATK_INTC | 2,092 / 10,000 | 0.000 | 0.346 | 0.006 | 0.585 | 0.433 | 0.083 |
| SCRIPTED-ATK_MSFT | 2,162 / 10,000 | 0.000 | 0.356 | 0.022 | 0.600 | 0.297 | 0.098 |
| SPOOFER-05_AMZN | 0 / 10,000 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## What this establishes

With attribution removed, the best learned model reaches step-level F1 **0.392** (logistic regression) and episode-level F1 **0.753** (logistic regression) on the held-out split, against **0.948** step-level F1 for the two features that were removed and 0.99+ order-level F1 for every attributed detector in the project.

### Against a constant classifier

| Formulation | Best learned model | Best constant classifier | Learned model wins? |
|---|---|---|---|
| Step | 0.392 (logistic regression) | 0.218 (constant: always flag) | **yes** |
| Episode | 0.753 (logistic regression) | 0.769 (constant: always flag) | **NO** |

**The episode-level F1 is not a result.** The held-out set is positive-heavy, so flagging every window scores a higher F1 than any model here. Episode-level F1 should not be quoted for this experiment; the step-level figures and the episode-level false-positive rates are the meaningful ones.

**This is the honest measure of the detection problem.** The attributed result was never evidence that manipulation is detectable; it was evidence that a label can be recovered from its own definition.

### Four things the breakdown shows that the headline F1 hides

1. **A sequence model is the best in-distribution and the worst out of it.** The supervised LSTM reaches step-level F1 0.86 and PR-AUC 0.93 in-distribution — far ahead of everything else — then collapses to F1 0.05 and PR-AUC 0.17 on held-out sources. It memorises the training sources' temporal signatures. Reporting only in-distribution numbers would have made it look like the clear winner.
2. **Detection only works on the narrow-spread stocks.** Per source, logistic regression reaches F1 0.585 on SCRIPTED-ATK_INTC and 0.600 on MSFT, but 0.073–0.131 on AAPL, GOOG and AMZN — where it **loses to a constant** (0.30–0.33). Attribution-free detection works where spoofing is profitable (1-tick stocks) and fails where it is not. The pooled F1 averages these opposite results into one number that describes neither.
3. **No model is calibrated out of distribution.** A classifier predicting the base rate scores Brier 0.099. On held-out, logistic regression scores 0.139, gradient boosting 0.110 and the LSTM 0.128 — all **worse**. In-distribution they beat it comfortably (0.029–0.110). These models rank better than chance but their probabilities should not be treated as probabilities.
4. **PR-AUC is the fairer headline than F1.** Against a held-out base rate of 0.096, logistic regression reaches 0.314 and gradient boosting 0.321 — roughly 3× the base rate, so there is real signal in the book alone, even though the F1 at a single threshold looks poor.

Models with ±0.000 intervals are deterministic given the data (the constants, the fitted rule, and logistic regression under lbfgs); only gradient boosting and the LSTM vary with the seed.

### What a false positive costs here

At step level the held-out positive rate is about 12%, so a false-positive rate near 0.08 means roughly one false alert for every two true ones. In a surveillance workflow that is an analyst queue where a third of items are noise — usable for triage, not for automated action.

### Limitation that bounds this experiment

In this environment only **one** participant acts on top of historical replay, so any anomaly in the displayed book is attributable to that participant by construction. A real anonymised feed mixes many participants, which makes the problem harder still. These numbers are therefore an **upper bound** on what attribution-free detection would achieve in a real market, not a lower bound.

