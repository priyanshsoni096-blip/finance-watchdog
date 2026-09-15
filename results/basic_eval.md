# Basic evaluation

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Episodes per run: 20. 95% CI uses a normal approximation (1.96·sd/√n). Runtime 678.8 s. The Watchdog is not built yet, so every detection number below is the **rule-based baseline** — the bar the Watchdog has to beat.

## C1 — Does RL learn meaningful manipulation?

PnL is also shown in units of (median spread × lot) so stocks are comparable.

| Agent | Group | Ticker | PnL mean ± 95% CI ($) | PnL (spread·lots) | Profitable eps | Trades/ep | Trades against own spoof/ep | Spoof orders/ep | Manipulative orders |
|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-01 | rl_train | AAPL | -254.5 ± 228.9 | -17.0 | 10% | 0.0 | 0.0 | 24.1 | 0% |
| SPOOFER-02 | rl_train | MSFT | +0.0 ± 0.0 | +0.0 | 0% | 0.0 | 0.0 | 0.0 | n/a% |
| SPOOFER-03 | rl_train | GOOG | -1,373.1 ± 1,041.6 | -50.9 | 5% | 0.0 | 0.0 | 33.1 | 0% |
| SPOOFER-04 | rl_train | INTC | +0.0 ± 0.0 | +0.0 | 0% | 0.0 | 0.0 | 0.0 | n/a% |
| SPOOFER-05 | rl_heldout | AMZN | -825.2 ± 578.0 | -63.5 | 10% | 0.0 | 0.0 | 5.0 | 0% |
| HONEST | honest_train | AAPL | -1,462.5 ± 214.4 | -97.5 | 0% | 198.0 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | AAPL | -206.1 ± 144.8 | -13.7 | 15% | 0.0 | 0.0 | 10.7 | 0% |
| SCRIPTED-ATK | scripted_train | AAPL | -1,021.2 ± 152.9 | -68.1 | 0% | 152.1 | 73.5 | 24.4 | 100% |
| HONEST | honest_train | MSFT | -8,193.7 ± 460.1 | -122.3 | 0% | 197.8 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | MSFT | -2,807.9 ± 1,888.9 | -41.9 | 5% | 0.0 | 0.0 | 11.0 | 0% |
| SCRIPTED-ATK | scripted_train | MSFT | -5,899.5 ± 1,451.5 | -88.1 | 0% | 146.3 | 72.1 | 23.6 | 100% |
| HONEST | honest_train | GOOG | -2,741.7 ± 267.4 | -101.5 | 0% | 201.6 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | GOOG | -538.3 ± 493.4 | -19.9 | 15% | 0.0 | 0.0 | 11.3 | 0% |
| SCRIPTED-ATK | scripted_train | GOOG | -1,819.7 ± 190.4 | -67.4 | 0% | 153.3 | 76.5 | 24.4 | 100% |
| HONEST | honest_train | INTC | -8,669.6 ± 553.7 | -122.1 | 0% | 198.6 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | INTC | -2,343.7 ± 2,127.0 | -33.0 | 5% | 0.0 | 0.0 | 10.6 | 0% |
| SCRIPTED-ATK | scripted_train | INTC | -4,990.6 ± 544.0 | -70.3 | 0% | 145.9 | 71.9 | 24.2 | 100% |
| HONEST | honest_heldout | AMZN | -1,185.3 ± 90.3 | -91.2 | 0% | 200.2 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_heldout | AMZN | -266.5 ± 146.4 | -20.5 | 0% | 0.0 | 0.0 | 10.8 | 0% |
| SCRIPTED-ATK | scripted_heldout | AMZN | -801.8 ± 53.3 | -61.7 | 0% | 140.6 | 69.0 | 22.8 | 100% |
| SPOOFER-04 (v1 policy, free accumulation) | exploit_replay | INTC | -24,332.7 ± 13,207.2 | -342.7 | 0% | 1332.6 | 1331.2 | 20.6 | 100% |
| SPOOFER-02 (v2 policy, alternating swing) | exploit_replay | MSFT | -8,051.6 ± 45,360.2 | -120.2 | 40% | 1918.8 | 1901.2 | 41.2 | 100% |
| SPOOFER-04 (v2 policy, alternating swing) | exploit_replay | INTC | -6,905.0 ± 37,103.4 | -97.3 | 60% | 1905.2 | 1893.0 | 40.2 | 100% |

## C2 / C3 — Rule-based baseline detector

Tuned on training stocks only: flag if size ≥ **2×** trailing touch depth and cancelled within **50** events without executing. An order is *manipulative* if its owner traded on the opposite side while it rested.

| Bucket | Orders | Manipulative | Legitimate | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|---|---|
| Tuning set (train stocks, all agents) | 3946 | 1931 | 2015 | 0.65 | 0.92 | 0.76 | 0.48 |
| RL training pool (AAPL/MSFT/GOOG/INTC) + flicker negatives | 2015 | 0 | 2015 | 0.00 | n/a | 0.00 | 0.48 |
| RL held-out SPOOFER-05 (AMZN) + flicker negatives | 314 | 0 | 314 | 0.00 | n/a | 0.00 | 0.18 |
| Scripted attacker, train stocks + flicker negatives | 2802 | 1931 | 871 | 0.93 | 0.92 | 0.93 | 0.14 |
| Scripted attacker, AMZN + flicker negatives | 670 | 455 | 215 | 0.92 | 0.91 | 0.92 | 0.16 |
| Flicker only (all orders legitimate) | 1086 | 0 | 1086 | 0.00 | n/a | 0.00 | 0.15 |

## C3 — Baseline on real, unlabeled LOBSTER order flow

Real data has no spoofing labels, so this is a flag *rate*, not a false-positive rate.

| Ticker | Events | New orders | Large orders (≥N× depth) | Flags | Flags per 10k events |
|---|---|---|---|---|---|
| AAPL | 370,391 | 176,867 | 2,037 | 1,153 | 31.13 |
| MSFT | 638,765 | 314,831 | 12 | 1 | 0.02 |
| GOOG | 117,916 | 56,666 | 308 | 162 | 13.74 |
| INTC | 594,040 | 290,056 | 32 | 9 | 0.15 |
| AMZN | 239,748 | 117,265 | 2,921 | 2,055 | 85.72 |
