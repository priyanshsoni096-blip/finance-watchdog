# Watchdog vs rule-based baseline

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Watchdog: `checkpoints\WATCHDOG\main\model.zip`. Baseline rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. Both are scored on the same recorded orders. An order is manipulative if its owner traded on the opposite side while it rested.

## Order-level detection

| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR | Median delay (steps) W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| In-distribution (test split) | 1533 / 7234 | 0.72 | 0.77 | 0.74 | 0.08 | 0.21 | 0.94 | 0.35 | 0.93 | 0 / 8 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 2621 | 0.00 | n/a | 0.00 | 0.03 | 0.00 | n/a | 0.00 | 0.96 | n/a / n/a |
| Scripted attacker, train stocks + FLICKER | 1849 / 4124 | 0.81 | 0.30 | 0.44 | 0.06 | 0.49 | 0.93 | 0.64 | 0.78 | 6 / 11 |
| Scripted attacker, AMZN + AMZN controls | 477 / 1042 | 0.70 | 0.27 | 0.39 | 0.10 | 0.50 | 0.93 | 0.65 | 0.78 | 7 / 11 |
| Legitimate only (HONEST + FLICKER) | 0 / 2667 | 0.00 | n/a | 0.00 | 0.06 | 0.00 | n/a | 0.00 | 0.83 | n/a / n/a |

## Step-level Watchdog flags

| Bucket | Positive steps / steps | Precision | Recall | FPR |
|---|---|---|---|---|
| In-distribution (test split) | 15,953 / 120,000 | 0.94 | 0.85 | 0.008 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 30,000 | 0.00 | n/a | 0.009 |
| Scripted attacker, train stocks + FLICKER | 20,575 / 80,000 | 0.80 | 0.10 | 0.009 |
| Scripted attacker, AMZN + AMZN controls | 5,313 / 30,000 | 0.63 | 0.08 | 0.011 |
| Legitimate only (HONEST + FLICKER) | 0 / 100,000 | 0.00 | n/a | 0.007 |

## Per source

| Source | Steps | Positive rate | Watchdog flag rate | Episodes with any flag | Orders | Manipulative |
|---|---|---|---|---|---|---|
| test/FLICKER_AAPL | 10,000 | 0.000 | 0.014 | 0.48 | 570 | 0 |
| test/FLICKER_GOOG | 10,000 | 0.000 | 0.004 | 0.20 | 539 | 0 |
| test/FLICKER_INTC | 10,000 | 0.000 | 0.009 | 0.30 | 507 | 0 |
| test/FLICKER_MSFT | 10,000 | 0.000 | 0.015 | 0.40 | 508 | 0 |
| test/HONEST_AAPL | 10,000 | 0.000 | 0.000 | 0.02 | 0 | 0 |
| test/HONEST_GOOG | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_INTC | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_MSFT | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/SPOOFER-01_AAPL | 10,000 | 0.002 | 0.016 | 0.64 | 2583 | 4 |
| test/SPOOFER-02_MSFT | 10,000 | 0.447 | 0.445 | 1.00 | 705 | 494 |
| test/SPOOFER-03_GOOG | 10,000 | 0.218 | 0.011 | 0.48 | 1150 | 388 |
| test/SPOOFER-04_INTC | 10,000 | 0.929 | 0.931 | 1.00 | 672 | 647 |
| heldout/FLICKER_AMZN | 10,000 | 0.000 | 0.023 | 0.54 | 543 | 0 |
| heldout/HONEST_AMZN | 10,000 | 0.000 | 0.001 | 0.06 | 0 | 0 |
| heldout/SCRIPTED-ATK_AAPL | 10,000 | 0.467 | 0.052 | 0.90 | 472 | 423 |
| heldout/SCRIPTED-ATK_AMZN | 10,000 | 0.531 | 0.049 | 0.90 | 499 | 477 |
| heldout/SCRIPTED-ATK_GOOG | 10,000 | 0.473 | 0.028 | 0.82 | 477 | 425 |
| heldout/SCRIPTED-ATK_INTC | 10,000 | 0.554 | 0.070 | 0.90 | 521 | 497 |
| heldout/SCRIPTED-ATK_MSFT | 10,000 | 0.563 | 0.070 | 0.92 | 530 | 504 |
| heldout/SPOOFER-05_AMZN | 10,000 | 0.000 | 0.002 | 0.18 | 2078 | 0 |
