# Is RL necessary? Supervised detectors on the same orders

The Watchdog learns from a reward and never sees a label. These classifiers are fitted directly on the hindsight labels, so they are an upper bound on what the 46 surveillance features support, not a competing contribution. Every model is scored on the same recorded orders, in the same buckets, with its decision threshold chosen on the train split alone.

Thresholds (fitted on train orders): logistic regression = 0.80 (train F1 0.59), gradient boosting = 0.95 (train F1 0.97)

## Order-level F1

| Bucket | Manipulative / orders | Rule | Logistic regression | Gradient boosting | RL Watchdog |
|---|---|---|---|---|---|
| In-distribution (test split) | 1543 / 7243 | 0.35 | 0.58 | 0.96 | 0.77 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 2920 | 0.00 | 0.00 | n/a | 0.00 |
| Scripted attacker, train stocks + FLICKER | 1942 / 4226 | 0.66 | 0.44 | 0.99 | 0.38 |
| Scripted attacker, AMZN + AMZN controls | 478 / 1042 | 0.65 | 0.21 | 0.99 | 0.26 |
| Legitimate only (HONEST + FLICKER) | 0 / 2688 | 0.00 | 0.00 | n/a | 0.00 |

## logistic regression — full order-level detail

| Bucket | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| In-distribution (test split) | 0.46 | 0.80 | 0.58 | 0.255 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0.00 | n/a | 0.00 | 0.125 |
| Scripted attacker, train stocks + FLICKER | 0.48 | 0.42 | 0.44 | 0.390 |
| Scripted attacker, AMZN + AMZN controls | 0.49 | 0.14 | 0.21 | 0.124 |
| Legitimate only (HONEST + FLICKER) | 0.00 | n/a | 0.00 | 0.348 |

## gradient boosting — full order-level detail

| Bucket | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| In-distribution (test split) | 0.92 | 1.00 | 0.96 | 0.025 |
| Held-out RL: SPOOFER-05 + AMZN controls | n/a | n/a | n/a | 0.000 |
| Scripted attacker, train stocks + FLICKER | 0.99 | 0.99 | 0.99 | 0.009 |
| Scripted attacker, AMZN + AMZN controls | 0.99 | 0.99 | 0.99 | 0.007 |
| Legitimate only (HONEST + FLICKER) | n/a | n/a | n/a | 0.000 |

## Reading this table

A supervised model that beats the RL Watchdog in-distribution shows the features carry the signal and the RL formulation is paying an optimisation cost for it. A supervised model that also collapses on the held-out attackers shows the generalisation failure is a property of the features and labels, not of reinforcement learning. The two conclusions are different and the table distinguishes them.

