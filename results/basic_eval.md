# Basic evaluation

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Episodes per run: 20. 95% CI uses a normal approximation (1.96·sd/√n). Runtime 54.2 s. Every detection number below is the **rule-based baseline** — the bar the Watchdog has to beat. The Watchdog is scored on the same kind of orders in `results/watchdog_eval.md`.

## C1 — Does RL learn meaningful manipulation?

PnL is also shown in units of (median spread × lot) so stocks are comparable.

| Agent | Group | Ticker | PnL mean ± 95% CI ($) | PnL (spread·lots) | Profitable eps | Trades/ep | Trades against own spoof/ep | Spoof orders/ep | Manipulative orders |
|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-01 | rl_train | AAPL | +61.3 ± 113.2 | +4.1 | 20% | 0.3 | 0.2 | 52.1 | 0% |
| SPOOFER-02 | rl_train | MSFT | +2,437.1 ± 1,067.6 | +36.4 | 95% | 108.8 | 58.7 | 13.9 | 70% |
| SPOOFER-03 | rl_train | GOOG | -19.3 ± 243.8 | -0.7 | 45% | 13.2 | 13.0 | 21.0 | 30% |
| SPOOFER-04 | rl_train | INTC | +13,208.1 ± 1,654.4 | +186.0 | 100% | 172.6 | 171.9 | 13.8 | 97% |
| SPOOFER-05 | rl_heldout | AMZN | -34.4 ± 65.1 | -2.6 | 0% | 0.1 | 0.0 | 45.9 | 0% |
| HONEST | honest_train | AAPL | -157.1 ± 29.2 | -10.5 | 0% | 21.6 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | AAPL | -404.8 ± 372.8 | -27.0 | 30% | 0.0 | 0.0 | 11.0 | 0% |
| SCRIPTED-ATK | scripted_train | AAPL | -440.2 ± 64.7 | -29.3 | 0% | 70.6 | 26.6 | 9.3 | 94% |
| HONEST | honest_train | MSFT | -971.3 ± 114.2 | -14.5 | 0% | 19.4 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | MSFT | -4,129.7 ± 3,810.0 | -61.6 | 0% | 0.0 | 0.0 | 10.5 | 0% |
| SCRIPTED-ATK | scripted_train | MSFT | +950.7 ± 859.9 | +14.2 | 65% | 65.0 | 29.9 | 10.5 | 96% |
| HONEST | honest_train | GOOG | -291.4 ± 55.8 | -10.8 | 0% | 19.2 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | GOOG | -707.4 ± 468.0 | -26.2 | 15% | 0.0 | 0.0 | 11.0 | 0% |
| SCRIPTED-ATK | scripted_train | GOOG | -636.6 ± 194.0 | -23.6 | 0% | 67.2 | 28.0 | 10.2 | 95% |
| HONEST | honest_train | INTC | -955.0 ± 120.5 | -13.5 | 0% | 20.4 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_train | INTC | -1,075.0 ± 1,136.0 | -15.1 | 0% | 0.0 | 0.0 | 11.4 | 0% |
| SCRIPTED-ATK | scripted_train | INTC | +629.4 ± 1,523.9 | +8.9 | 80% | 62.0 | 28.7 | 10.7 | 95% |
| HONEST | honest_heldout | AMZN | -133.6 ± 31.7 | -10.3 | 0% | 20.2 | 0.0 | 0.0 | n/a% |
| FLICKER | flicker_heldout | AMZN | -320.8 ± 131.3 | -24.7 | 0% | 0.0 | 0.0 | 11.3 | 0% |
| SCRIPTED-ATK | scripted_heldout | AMZN | -308.0 ± 70.1 | -23.7 | 0% | 62.3 | 29.1 | 10.3 | 95% |
| SPOOFER-04 (v1 policy, free accumulation) | exploit_replay | INTC | -24,332.7 ± 13,207.2 | -342.7 | 0% | 1332.6 | 1331.2 | 20.6 | 100% |
| SPOOFER-02 (v2 policy, alternating swing) | exploit_replay | MSFT | -8,051.6 ± 45,360.2 | -120.2 | 40% | 1918.8 | 1901.2 | 41.2 | 100% |
| SPOOFER-04 (v2 policy, alternating swing) | exploit_replay | INTC | -6,905.0 ± 37,103.4 | -97.3 | 60% | 1905.2 | 1893.0 | 40.2 | 100% |

## C2 / C3 — Rule-based baseline detector

Tuned on training stocks only: flag if size ≥ **2×** trailing touch depth and cancelled within **200** events without executing. An order is *manipulative* if its owner traded on the opposite side while it rested.

| Bucket | Orders | Manipulative | Legitimate | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|---|---|
| Tuning set (train stocks, all agents) | 3711 | 1366 | 2345 | 0.38 | 0.88 | 0.53 | 0.83 |
| RL training pool (AAPL/MSFT/GOOG/INTC) + flicker negatives | 2895 | 591 | 2304 | 0.20 | 0.82 | 0.32 | 0.84 |
| RL held-out SPOOFER-05 (AMZN) + flicker negatives | 1144 | 0 | 1144 | 0.00 | n/a | 0.00 | 0.93 |
| Scripted attacker, train stocks + flicker negatives | 1694 | 775 | 919 | 0.57 | 0.93 | 0.71 | 0.59 |
| Scripted attacker, AMZN + flicker negatives | 432 | 196 | 236 | 0.56 | 0.97 | 0.71 | 0.63 |
| Flicker only (all orders legitimate) | 1104 | 0 | 1104 | 0.00 | n/a | 0.00 | 0.62 |

## C3 — Baseline on real, unlabeled LOBSTER order flow

Real data has no spoofing labels, so this is a flag *rate*, not a false-positive rate.

| Ticker | Events | New orders | Large orders (≥N× depth) | Flags | Flags per 10k events |
|---|---|---|---|---|---|
| AAPL | 370,391 | 176,867 | 2,037 | 1,322 | 35.69 |
| MSFT | 638,765 | 314,831 | 12 | 2 | 0.03 |
| GOOG | 117,916 | 56,666 | 308 | 209 | 17.72 |
| INTC | 594,040 | 290,056 | 32 | 17 | 0.29 |
| AMZN | 239,748 | 117,265 | 2,921 | 2,443 | 101.90 |
