# Pre-registered held-out test: REVERSAL-ATK

Defined, committed and pushed before any of its data or results existed; nothing was changed after seeing them.

**H1:** Watchdog order-level recall on REVERSAL-ATK >= 0.8. **Result: supported** (Watchdog recall 0.981; rule 0.964).

**H2:** Watchdog order-level FPR on REVERT + FLICKER <= 0.1. **Result: supported** (Watchdog FPR 0.006; rule 0.804).

**H3:** Watchdog order-level F1 on REVERSAL-ATK + REVERT + FLICKER >= rule F1. **Result: supported** (Watchdog F1 0.986, rule F1 0.591).

Verdict model `checkpoints/WATCHDOG/cov`. Rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. 50 episodes per stock of each source.

| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|
| Watchdog: All sources | 2504 / 6849 | 2456 / 24 / 48 / 4321 | 0.99 | 0.98 | 0.99 | 0.01 |
| Rule: All sources | 2504 / 6849 | 2415 / 3259 / 89 / 1086 | 0.43 | 0.96 | 0.59 | 0.75 |
| Watchdog: REVERSAL-ATK only | 2504 / 2797 | 2456 / 0 / 48 / 293 | 1.00 | 0.98 | 0.99 | 0.00 |
| Rule: REVERSAL-ATK only | 2504 / 2797 | 2415 / 3 / 89 / 290 | 1.00 | 0.96 | 0.98 | 0.01 |
| Watchdog: REVERT + FLICKER | 0 / 4052 | 0 / 24 / 0 / 4028 | 0.00 | n/a | 0.00 | 0.01 |
| Rule: REVERT + FLICKER | 0 / 4052 | 0 / 3256 / 0 / 796 | 0.00 | n/a | 0.00 | 0.80 |
| Watchdog: REVERT only | 0 / 1360 | 0 / 21 / 0 / 1339 | 0.00 | n/a | 0.00 | 0.02 |
| Rule: REVERT only | 0 / 1360 | 0 / 1051 / 0 / 309 | 0.00 | n/a | 0.00 | 0.77 |
| Watchdog: FLICKER only | 0 / 2692 | 0 / 3 / 0 / 2689 | 0.00 | n/a | 0.00 | 0.00 |
| Rule: FLICKER only | 0 / 2692 | 0 / 2205 / 0 / 487 | 0.00 | n/a | 0.00 | 0.82 |

Per stock, verdict model (context only):

| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|
| Watchdog: AAPL | 483 / 1392 | 466 / 5 / 17 / 904 | 0.99 | 0.96 | 0.98 | 0.01 |
| Watchdog: MSFT | 499 / 1336 | 494 / 10 / 5 / 827 | 0.98 | 0.99 | 0.99 | 0.01 |
| Watchdog: GOOG | 483 / 1388 | 474 / 1 / 9 / 904 | 1.00 | 0.98 | 0.99 | 0.00 |
| Watchdog: INTC | 522 / 1356 | 512 / 4 / 10 / 830 | 0.99 | 0.98 | 0.99 | 0.00 |
| Watchdog: AMZN | 517 / 1377 | 510 / 4 / 7 / 856 | 0.99 | 0.99 | 0.99 | 0.00 |

Other models on the same orders (context only, not used for any verdict):

| Model | Pooled F1 | REVERSAL-ATK recall | REVERT FPR | FLICKER FPR |
|---|---|---|---|---|
| cov | 0.986 | 0.981 | 0.015 | 0.001 |
| cov_s2 | 0.981 | 0.984 | 0.034 | 0.003 |
| cov_s3 | 0.977 | 0.984 | 0.049 | 0.003 |
| causal_g0_1m | 0.570 | 0.400 | 0.005 | 0.001 |
| main | 0.198 | 0.121 | 0.072 | 0.045 |

REVERSAL-ATK PnL per episode (context only): AAPL -392, MSFT -657, GOOG -700, INTC +258, AMZN -217
