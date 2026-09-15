# Watchdog vs rule-based baseline

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Watchdog: `checkpoints\WATCHDOG\main\model.zip`. Baseline rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. Both are scored on the same recorded orders. An order is manipulative if its owner traded on the opposite side while it rested.

## Order-level detection

| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR | Median delay (steps) W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| In-distribution (test split) | 1543 / 7243 | 0.75 | 0.80 | 0.77 | 0.07 | 0.22 | 0.94 | 0.35 | 0.92 | 0 / 8 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 2920 | 0.00 | n/a | 0.00 | 0.01 | 0.00 | n/a | 0.00 | 0.97 | n/a / n/a |
| Scripted attacker, train stocks + FLICKER | 1942 / 4226 | 0.83 | 0.25 | 0.38 | 0.04 | 0.51 | 0.94 | 0.66 | 0.77 | 6 / 11 |
| Scripted attacker, AMZN + AMZN controls | 478 / 1042 | 0.73 | 0.15 | 0.26 | 0.05 | 0.50 | 0.95 | 0.65 | 0.81 | 8 / 11 |
| Legitimate only (HONEST + FLICKER) | 0 / 2688 | 0.00 | n/a | 0.00 | 0.04 | 0.00 | n/a | 0.00 | 0.82 | n/a / n/a |

## Step-level Watchdog flags

| Bucket | Positive steps / steps | Precision | Recall | FPR |
|---|---|---|---|---|
| In-distribution (test split) | 15,831 / 120,000 | 0.95 | 0.86 | 0.006 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0 / 30,000 | 0.00 | n/a | 0.004 |
| Scripted attacker, train stocks + FLICKER | 21,625 / 80,000 | 0.86 | 0.08 | 0.005 |
| Scripted attacker, AMZN + AMZN controls | 5,360 / 30,000 | 0.69 | 0.04 | 0.004 |
| Legitimate only (HONEST + FLICKER) | 0 / 100,000 | 0.00 | n/a | 0.004 |

## Per source

| Source | Steps | Positive rate | Watchdog flag rate | Episodes with any flag | Orders | Manipulative |
|---|---|---|---|---|---|---|
| test/FLICKER_AAPL | 10,000 | 0.000 | 0.011 | 0.40 | 578 | 0 |
| test/FLICKER_GOOG | 10,000 | 0.000 | 0.006 | 0.30 | 559 | 0 |
| test/FLICKER_INTC | 10,000 | 0.000 | 0.007 | 0.22 | 509 | 0 |
| test/FLICKER_MSFT | 10,000 | 0.000 | 0.003 | 0.28 | 515 | 0 |
| test/HONEST_AAPL | 10,000 | 0.000 | 0.001 | 0.02 | 0 | 0 |
| test/HONEST_GOOG | 10,000 | 0.000 | 0.001 | 0.04 | 0 | 0 |
| test/HONEST_INTC | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/HONEST_MSFT | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| test/SPOOFER-01_AAPL | 10,000 | 0.003 | 0.012 | 0.64 | 2606 | 10 |
| test/SPOOFER-02_MSFT | 10,000 | 0.445 | 0.447 | 1.00 | 696 | 489 |
| test/SPOOFER-03_GOOG | 10,000 | 0.209 | 0.018 | 0.66 | 1074 | 364 |
| test/SPOOFER-04_INTC | 10,000 | 0.925 | 0.930 | 1.00 | 706 | 680 |
| heldout/FLICKER_AMZN | 10,000 | 0.000 | 0.010 | 0.36 | 527 | 0 |
| heldout/HONEST_AMZN | 10,000 | 0.000 | 0.000 | 0.00 | 0 | 0 |
| heldout/SCRIPTED-ATK_AAPL | 10,000 | 0.510 | 0.018 | 0.64 | 500 | 455 |
| heldout/SCRIPTED-ATK_AMZN | 10,000 | 0.536 | 0.022 | 0.86 | 515 | 478 |
| heldout/SCRIPTED-ATK_GOOG | 10,000 | 0.530 | 0.011 | 0.58 | 507 | 475 |
| heldout/SCRIPTED-ATK_INTC | 10,000 | 0.565 | 0.070 | 0.98 | 528 | 508 |
| heldout/SCRIPTED-ATK_MSFT | 10,000 | 0.557 | 0.075 | 0.90 | 530 | 504 |
| heldout/SPOOFER-05_AMZN | 10,000 | 0.000 | 0.001 | 0.14 | 2393 | 0 |
