# Detection without participant attribution

Every other detector in this project is handed the monitored participant's own trade direction and the side of its resting orders — together, the manipulation label's definition. This experiment removes that attribution entirely and uses **only the displayed order book**, as an external surveillance system on an anonymised feed would see it.

Data: `C:\Users\Rahul Soni\Desktop\finance-watchdog\data\watchdog_v2`. Splits are disjoint stretches of the trading day where available; preprocessing and every threshold are fitted on training data only; episodes never cross splits.

## Leakage audit of the retained features

| Check | Value |
|---|---|
| Steps in train | 360,000 |
| Positive rate (causal label) | 0.111 |
| **Withheld pair** (features 44 + 45) step F1 — what we removed | **0.948** |
| Largest absolute correlation of any retained feature with the label | 0.381 |
| Retained features with correlation > 0.5 | 0 |
| Best single retained feature threshold rule (swept on train) | 0.490 (`total displayed depth`) |
| Most label-correlated retained feature | `book L1 bid size` |

The withheld pair reconstructs the label almost perfectly; no retained feature comes close. The removal is therefore real — the label is not simply reachable by another column.

## Step level — is manipulation happening now?

| Model | Split | Precision | Recall | F1 | FPR | Positives / n |
|---|---|---|---|---|---|---|
| constant: never flag | test (in-distribution) | n/a | 0.000 | 0.000 | 0.000 | 13,866 / 120,000 |
| constant: never flag | held-out | n/a | 0.000 | 0.000 | 0.000 | 9,781 / 80,000 |
| constant: always flag | test (in-distribution) | 0.116 | 1.000 | 0.207 | 1.000 | 13,866 / 120,000 |
| constant: always flag | held-out | 0.122 | 1.000 | 0.218 | 1.000 | 9,781 / 80,000 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | test (in-distribution) | 0.046 | 0.029 | 0.036 | 0.079 | 13,866 / 120,000 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | held-out | 0.120 | 0.085 | 0.099 | 0.086 | 9,781 / 80,000 |
| logistic regression | test (in-distribution) | 0.491 | 0.851 | 0.622 | 0.115 | 13,866 / 120,000 |
| logistic regression | held-out | 0.409 | 0.376 | 0.392 | 0.075 | 9,781 / 80,000 |
| gradient boosting | test (in-distribution) | 0.605 | 0.586 | 0.595 | 0.050 | 13,866 / 120,000 |
| gradient boosting | held-out | 0.533 | 0.125 | 0.202 | 0.015 | 9,781 / 80,000 |

## Episode level — does this window contain manipulation?

| Model | Split | Precision | Recall | F1 | FPR | Positives / n |
|---|---|---|---|---|---|---|
| constant: never flag | test (in-distribution) | n/a | 0.000 | 0.000 | 0.000 | 154 / 600 |
| constant: never flag | held-out | n/a | 0.000 | 0.000 | 0.000 | 250 / 400 |
| constant: always flag | test (in-distribution) | 0.257 | 1.000 | 0.408 | 1.000 | 154 / 600 |
| constant: always flag | held-out | 0.625 | 1.000 | 0.769 | 1.000 | 250 / 400 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | test (in-distribution) | 0.183 | 0.357 | 0.242 | 0.552 | 154 / 600 |
| book-imbalance rule (|deep imbalance| > p90 = 0.279) | held-out | 0.515 | 0.624 | 0.564 | 0.980 | 250 / 400 |
| logistic regression | test (in-distribution) | 0.380 | 0.974 | 0.546 | 0.549 | 154 / 600 |
| logistic regression | held-out | 0.685 | 0.836 | 0.753 | 0.640 | 250 / 400 |
| gradient boosting | test (in-distribution) | 0.539 | 0.890 | 0.672 | 0.262 | 154 / 600 |
| gradient boosting | held-out | 0.981 | 0.416 | 0.584 | 0.013 | 250 / 400 |

## What this establishes

With attribution removed, the best learned model reaches step-level F1 **0.392** (logistic regression) and episode-level F1 **0.753** (logistic regression) on the held-out split, against **0.948** step-level F1 for the two features that were removed and 0.99+ order-level F1 for every attributed detector in the project.

### Against a constant classifier

| Formulation | Best learned model | Best constant classifier | Learned model wins? |
|---|---|---|---|
| Step | 0.392 (logistic regression) | 0.218 (constant: always flag) | **yes** |
| Episode | 0.753 (logistic regression) | 0.769 (constant: always flag) | **NO** |

**The episode-level F1 is not a result.** The held-out set is positive-heavy, so flagging every window scores a higher F1 than any model here. Episode-level F1 should not be quoted for this experiment; the step-level figures and the episode-level false-positive rates are the meaningful ones. Gradient boosting is the only model that separates anything usefully at episode level, and it does so by being conservative (low recall, very low FPR), not by beating the constant on F1.

**This is the honest measure of the detection problem.** The attributed result was never evidence that manipulation is detectable; it was evidence that a label can be recovered from its own definition.

### Limitation that bounds this experiment

In this environment only **one** participant acts on top of historical replay, so any anomaly in the displayed book is attributable to that participant by construction. A real anonymised feed mixes many participants, which makes the problem harder still. These numbers are therefore an **upper bound** on what attribution-free detection would achieve in a real market, not a lower bound.

