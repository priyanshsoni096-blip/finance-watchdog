# Basic evaluation

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Episodes per run: 20. 95% CI uses a normal approximation (1.96·sd/√n). Runtime 394.4 s. The Watchdog is not built yet, so every detection number below is the **rule-based baseline** — the bar the Watchdog has to beat.

## C1 — Does RL learn meaningful manipulation?

PnL is also shown in units of (median spread × lot) so stocks are comparable.

| Agent | Group | Ticker | PnL mean ± 95% CI ($) | PnL (spread·lots) | Profitable eps | Trades/ep | Trades against own spoof/ep | Spoof orders/ep | Manipulative orders |
|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-01 | rl_train | AAPL | +0.0 ± 0.0 | +0.0 | 0% | 0.0 | 0.0 | 9.2 | 0% |
| SPOOFER-02 | rl_train | MSFT | +238,519.3 ± 6,686.9 | +3560.0 | 100% | 1918.0 | 1900.0 | 41.4 | 100% |
| SPOOFER-03 | rl_train | GOOG | +66.8 ± 189.6 | +2.5 | 40% | 7.6 | 7.6 | 14.3 | 11% |
| SPOOFER-04 | rl_train | INTC | +249,446.5 ± 4,538.1 | +3513.3 | 100% | 1910.0 | 1901.5 | 40.8 | 100% |
| SPOOFER-05 | rl_heldout | AMZN | -360.4 ± 145.8 | -27.7 | 5% | 0.0 | 0.0 | 8.4 | 0% |
| HONEST | honest_train | AAPL | -1,462.5 ± 214.4 | -97.5 | 0% | 198.0 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | AAPL | -206.1 ± 144.8 | -13.7 | 15% | 0.0 | 0.0 | 10.7 | 0% |
| SCRIPTED-ATK | scripted_train | AAPL | -650.2 ± 110.4 | -43.3 | 0% | 152.1 | 73.5 | 24.4 | 100% |
| HONEST | honest_train | MSFT | -8,193.7 ± 460.1 | -122.3 | 0% | 197.8 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | MSFT | -2,807.9 ± 1,888.9 | -41.9 | 5% | 0.0 | 0.0 | 11.0 | 0% |
| SCRIPTED-ATK | scripted_train | MSFT | +4,776.9 ± 1,322.3 | +71.3 | 90% | 146.3 | 72.1 | 23.6 | 100% |
| HONEST | honest_train | GOOG | -2,741.7 ± 267.4 | -101.5 | 0% | 201.6 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | GOOG | -538.3 ± 493.4 | -19.9 | 15% | 0.0 | 0.0 | 11.3 | 0% |
| SCRIPTED-ATK | scripted_train | GOOG | -792.0 ± 121.9 | -29.3 | 0% | 153.3 | 76.5 | 24.4 | 100% |
| HONEST | honest_train | INTC | -8,669.6 ± 553.7 | -122.1 | 0% | 198.6 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | INTC | -2,343.7 ± 2,127.0 | -33.0 | 5% | 0.0 | 0.0 | 10.6 | 0% |
| SCRIPTED-ATK | scripted_train | INTC | +6,028.9 ± 818.7 | +84.9 | 100% | 145.9 | 71.9 | 24.2 | 100% |
| HONEST | honest_heldout | AMZN | -1,185.3 ± 90.3 | -91.2 | 0% | 200.2 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_heldout | AMZN | -266.5 ± 146.4 | -20.5 | 0% | 0.0 | 0.0 | 10.8 | 0% |
| SCRIPTED-ATK | scripted_heldout | AMZN | -531.5 ± 45.8 | -40.9 | 0% | 140.6 | 69.0 | 22.8 | 100% |
| SPOOFER-04 (pre-fix policy) | exploit_replay | INTC | +171,192.0 ± 41,029.1 | +2411.2 | 100% | 1318.0 | 1315.8 | 20.4 | 100% |

## C2 / C3 — Rule-based baseline detector

Tuned on training stocks only: flag if size ≥ **2×** trailing touch depth and cancelled within **200** events without executing. An order is *manipulative* if its owner traded on the opposite side while it rested.

| Bucket | Orders | Manipulative | Legitimate | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|---|---|
| Tuning set (train stocks, all agents) | 4915 | 3605 | 1310 | 0.79 | 0.97 | 0.87 | 0.72 |
| RL training pool (AAPL/MSFT/GOOG/INTC) + flicker negatives | 2984 | 1674 | 1310 | 0.63 | 0.97 | 0.77 | 0.72 |
| RL held-out SPOOFER-05 (AMZN) + flicker negatives | 384 | 0 | 384 | 0.00 | n/a | 0.00 | 0.61 |
| Scripted attacker, train stocks + flicker negatives | 2802 | 1931 | 871 | 0.78 | 0.97 | 0.87 | 0.60 |
| Scripted attacker, AMZN + flicker negatives | 670 | 455 | 215 | 0.78 | 0.97 | 0.86 | 0.60 |
| Flicker only (all orders legitimate) | 1086 | 0 | 1086 | 0.00 | n/a | 0.00 | 0.60 |

## C3 — Baseline on real, unlabeled LOBSTER order flow

Real data has no spoofing labels, so this is a flag *rate*, not a false-positive rate.

| Ticker | Events | New orders | Large orders (≥N× depth) | Flags | Flags per 10k events |
|---|---|---|---|---|---|
| AAPL | 370,391 | 176,867 | 2,037 | 1,322 | 35.69 |
| MSFT | 638,765 | 314,831 | 12 | 2 | 0.03 |
| GOOG | 117,916 | 56,666 | 308 | 209 | 17.72 |
| INTC | 594,040 | 290,056 | 32 | 17 | 0.29 |
| AMZN | 239,748 | 117,265 | 2,921 | 2,443 | 101.90 |
