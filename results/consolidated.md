# Consolidated results: Watchdog vs rule-based baseline

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Built by `scripts/consolidate_results.py` from the committed `results/*.json`; no numbers are re-estimated. Main Watchdog (`checkpoints/WATCHDOG/main`) vs the rule tuned on the replay train split (size ≥ 2× depth, cancelled within 1000 events). Order level. An order is manipulative if its owner traded on the opposite side while it rested. Delays are agent steps of 10 market events from placement to first flag.

## Detection, every bucket

Confusion matrix is TP / FP / FN / TN.

| Market | Bucket | Watchdog TP/FP/FN/TN | Watchdog P | R | F1 | FPR | Rule TP/FP/FN/TN | Rule P | R | F1 | FPR | Median delay W / R | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| replay | In-distribution (test split) | 1230 / 414 / 313 / 5286 | 0.75 | 0.80 | 0.77 | 0.07 | 1452 / 5244 / 91 / 456 | 0.22 | 0.94 | 0.35 | 0.92 | 0 / 8 | `watchdog_eval.json` |
| replay | Held-out RL: SPOOFER-05 + AMZN controls | 0 / 37 / 0 / 2883 | 0.00 | n/a | 0.00 | 0.01 | 0 / 2837 / 0 / 83 | 0.00 | n/a | 0.00 | 0.97 | n/a / n/a | `watchdog_eval.json` |
| replay | Scripted attacker, train stocks + FLICKER | 476 / 98 / 1466 / 2186 | 0.83 | 0.25 | 0.38 | 0.04 | 1821 / 1754 / 121 / 530 | 0.51 | 0.94 | 0.66 | 0.77 | 6 / 11 | `watchdog_eval.json` |
| replay | Scripted attacker, AMZN + AMZN controls | 74 / 27 / 404 / 537 | 0.73 | 0.15 | 0.26 | 0.05 | 454 / 458 / 24 / 106 | 0.50 | 0.95 | 0.65 | 0.81 | 8 / 11 | `watchdog_eval.json` |
| replay | Legitimate only (HONEST + FLICKER) | 0 / 120 / 0 / 2568 | 0.00 | n/a | 0.00 | 0.04 | 0 / 2212 / 0 / 476 | 0.00 | n/a | 0.00 | 0.82 | n/a / n/a | `watchdog_eval.json` |
| replay | LATEBURST-ATK (pre-registered) + FLICKER, all stocks | 271 / 140 / 861 / 2772 | 0.66 | 0.24 | 0.35 | 0.05 | 1002 / 2205 / 130 / 707 | 0.31 | 0.89 | 0.46 | 0.76 | 12 / 22 | `lateburst_test.json` |
| synthetic | Manipulators + legitimate | 543 / 77 / 582 / 691 | 0.88 | 0.48 | 0.62 | 0.10 | 1085 / 739 / 40 / 29 | 0.59 | 0.96 | 0.74 | 0.96 | 0 / 7 | `synthetic_eval.json` |
| synthetic | Legitimate only | 0 / 34 / 0 / 268 | 0.00 | n/a | 0.00 | 0.11 | 0 / 277 / 0 / 25 | 0.00 | n/a | 0.00 | 0.92 | n/a / n/a | `synthetic_eval.json` |
| synthetic | LAYER-ATK (pre-registered) + FLICKER | 120 / 24 / 280 / 294 | 0.83 | 0.30 | 0.44 | 0.08 | 400 / 291 / 0 / 27 | 0.58 | 1.00 | 0.73 | 0.92 | 0 / 9 | `layering_test.json` |

## Three Watchdog seeds (mean ± 95% Student-t CI)

Seeds: main, main_s2, main_s3. Source: `multiseed.json`.

| Metric | Watchdog | Rule |
|---|---|---|
| in-distribution order F1 | 0.753 ± 0.058 | 0.352 |
| in-distribution order precision | 0.821 ± 0.277 | 0.217 |
| in-distribution order recall | 0.715 ± 0.274 | 0.941 |
| in-distribution step precision | 0.966 ± 0.060 | n/a |
| in-distribution step recall | 0.743 ± 0.522 | n/a |
| legitimate-only order FPR | 0.031 ± 0.067 | 0.823 |
| held-out AMZN legitimate order FPR | 0.004 ± 0.018 | 0.972 |
| LATEBURST-ATK order recall | 0.217 ± 0.156 | 0.885 |
| LATEBURST-ATK order precision | 0.774 ± 0.486 | 0.312 |
| LATEBURST-ATK order FPR | 0.034 ± 0.072 | 0.757 |

## PnL suppression (main Watchdog, surveillance acting inside episodes)

Mean PnL per episode ± 95% CI; suppression only meaningful where PnL without surveillance is clearly positive. Source: `pnl_suppression.json`.

| Source | Stock | Kind | No surveillance | Watchdog | Suppression | Rule | Suppression |
|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | manipulator | +3,186 ± 517 | -4,560 ± 1,275 | 243% | +140 ± 830 | 96% |
| SPOOFER-04 | INTC | manipulator | +12,625 ± 1,162 | +2,165 ± 485 | 83% | +5,655 ± 657 | 55% |
| SCRIPTED-ATK | MSFT | manipulator | +418 ± 956 | -548 ± 594 | 231% | -945 ± 371 | 326% |
| LATEBURST-ATK | MSFT | manipulator | +531 ± 674 | -693 ± 723 | 231% | -974 ± 757 | 283% |
| HONEST | MSFT | legitimate | -961 ± 101 | -961 ± 101 | — | -961 ± 101 | — |
| FLICKER | MSFT | legitimate | -2,481 ± 1,026 | -3,089 ± 1,247 | — | -2,045 ± 990 | — |
| SCRIPTED-ATK | INTC | manipulator | +748 ± 634 | -20 ± 525 | 103% | -1,281 ± 613 | 271% |
| LATEBURST-ATK | INTC | manipulator | -87 ± 1,007 | +262 ± 622 | — | -86 ± 527 | — |
| HONEST | INTC | legitimate | -875 ± 78 | -875 ± 78 | — | -875 ± 78 | — |
| FLICKER | INTC | legitimate | -2,094 ± 1,114 | -1,883 ± 1,103 | — | -1,267 ± 677 | — |

## Rule on real, unlabeled LOBSTER order flow

No labels exist, so this is a flag rate, not a false-positive rate. The Watchdog cannot run here: LOBSTER has no participant identities. Source: `basic_eval.json`.

| Stock | Events | Large orders | Flags | Flags per 10k events |
|---|---|---|---|---|
| AAPL | 370,391 | 2,037 | 1,322 | 35.69 |
| MSFT | 638,765 | 12 | 2 | 0.03 |
| GOOG | 117,916 | 308 | 209 | 17.72 |
| INTC | 594,040 | 32 | 17 | 0.29 |
| AMZN | 239,748 | 2,921 | 2,443 | 101.90 |

## Sensitivity: rule re-tuned on each bucket's own orders (oracle upper bound)

Requested by the execution plan's risk register. Each bucket's rule parameters are chosen with hindsight on the very orders it is scored on, so this is the best any (N, M) rule could do, not a deployable detector. It does not change any pre-registered verdict. Buckets with no manipulative orders have no F1, so their parameters are just the grid's most conservative choice.

| Bucket | Oracle N × depth | Oracle M events | TP/FP/FN/TN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|---|
| In-distribution (test split) | 2 | 1000 | 1452 / 5244 / 91 / 456 | 0.22 | 0.94 | 0.35 | 0.92 |
| Held-out RL: SPOOFER-05 + AMZN controls | 2 | 10 | 0 / 2131 / 0 / 789 | 0.00 | n/a | 0.00 | 0.73 |
| Scripted attacker, train stocks + FLICKER | 2 | 200 | 1821 / 1257 / 121 / 1027 | 0.59 | 0.94 | 0.73 | 0.55 |
| Scripted attacker, AMZN + AMZN controls | 2 | 200 | 454 / 327 / 24 / 237 | 0.58 | 0.95 | 0.72 | 0.58 |
| Legitimate only (HONEST + FLICKER) | 2 | 10 | 0 / 94 / 0 / 2594 | 0.00 | n/a | 0.00 | 0.03 |
| LATEBURST-ATK (pre-registered) + FLICKER, all stocks | 2 | 1000 | 1002 / 2205 / 130 / 707 | 0.31 | 0.89 | 0.46 | 0.76 |
