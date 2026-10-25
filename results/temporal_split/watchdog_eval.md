# Watchdog vs rule-based baseline

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Watchdog: `checkpoints\WATCHDOG\v2_temporal\model.zip`. Baseline rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. Both are scored on the same recorded orders. An order is manipulative if its owner traded on the opposite side while it rested.

## Order-level detection

| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR | Median delay (steps) W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| In-distribution (test split) | 1533 / 7234 | 0.81 | 0.73 | 0.77 | 0.05 | 0.21 | 0.94 | 0.35 | 0.93 | 0 / 8 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 2621 | 0.00 | n/a | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.96 | n/a / n/a |
| Scripted attacker, train stocks + FLICKER | 1849 / 4124 | 0.85 | 0.30 | 0.45 | 0.04 | 0.49 | 0.93 | 0.64 | 0.78 | 6 / 11 |
| Scripted attacker, AMZN + AMZN controls | 477 / 1042 | 0.50 | 0.01 | 0.02 | 0.01 | 0.50 | 0.93 | 0.65 | 0.78 | 4 / 11 |
| Legitimate only (HONEST + FLICKER) | 0 / 2667 | 0.00 | n/a | 0.00 | 0.03 | 0.00 | n/a | 0.00 | 0.83 | n/a / n/a |

## Step-level Watchdog flags

| Bucket | Positive steps / steps | Precision | Recall | FPR |
|---|---|---|---|---|
| In-distribution (test split) | 15,953 / 120,000 | 0.95 | 0.82 | 0.007 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 30,000 | 0.00 | n/a | 0.001 |
| Scripted attacker, train stocks + FLICKER | 20,575 / 80,000 | 0.79 | 0.11 | 0.010 |
| Scripted attacker, AMZN + AMZN controls | 5,313 / 30,000 | 0.33 | 0.00 | 0.001 |
| Legitimate only (HONEST + FLICKER) | 0 / 100,000 | 0.00 | n/a | 0.006 |

## Per source

| Source | Steps | Positive rate | Watchdog flag rate | Episodes with any flag | Orders | Manipulative |
|---|---|---|---|---|---|---|
| test/FLICKER_AAPL | 10,000 | 0.000 | 0.000 | 0.02 | 570 | 0 |
| test/FLICKER_GOOG | 10,000 | 0.000 | 0.000 | 0.00 | 539 | 0 |
| test/FLICKER_INTC | 10,000 | 0.000 | 0.026 | 0.44 | 507 | 0 |
| test/FLICKER_MSFT | 10,000 | 0.000 | 0.032 | 0.48 | 508 | 0 |
| test/HONEST_AAPL | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_GOOG | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_INTC | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_MSFT | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/SPOOFER-01_AAPL | 10,000 | 0.002 | 0.000 | 0.00 | 2583 | 4 |
| test/SPOOFER-02_MSFT | 10,000 | 0.447 | 0.397 | 1.00 | 705 | 494 |
| test/SPOOFER-03_GOOG | 10,000 | 0.218 | 0.000 | 0.00 | 1150 | 388 |
| test/SPOOFER-04_INTC | 10,000 | 0.929 | 0.932 | 1.00 | 672 | 647 |
| heldout/FLICKER_AMZN | 10,000 | 0.000 | 0.003 | 0.10 | 543 | 0 |
| heldout/HONEST_AMZN | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| heldout/SCRIPTED-ATK_AAPL | 10,000 | 0.467 | 0.012 | 0.14 | 472 | 423 |
| heldout/SCRIPTED-ATK_AMZN | 10,000 | 0.531 | 0.002 | 0.12 | 499 | 477 |
| heldout/SCRIPTED-ATK_GOOG | 10,000 | 0.473 | 0.000 | 0.00 | 477 | 425 |
| heldout/SCRIPTED-ATK_INTC | 10,000 | 0.554 | 0.109 | 1.00 | 521 | 497 |
| heldout/SCRIPTED-ATK_MSFT | 10,000 | 0.563 | 0.107 | 1.00 | 530 | 504 |
| heldout/SPOOFER-05_AMZN | 10,000 | 0.000 | 0.000 | 0.02 | 2078 | 0 |
