# Pre-registered held-out test: DRIP-ATK

Defined and committed before any of its data or results existed; nothing was changed after seeing them.

**H1:** Watchdog order-level F1 on DRIP-ATK + SAMESIDE + FLICKER >= rule F1. **Result: supported** (Watchdog F1 0.409, rule F1 0.336).

**H2:** Watchdog order-level FPR on SAMESIDE <= 0.1. **Result: supported** (SAMESIDE FPR 0.012).

Verdict model `checkpoints/WATCHDOG/causal_g0_1m`. Rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. 50 episodes per stock of each source.

| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|
| Watchdog: All sources | 1218 / 5413 | 318 / 19 / 900 / 4176 | 0.94 | 0.26 | 0.41 | 0.00 |
| Rule: All sources | 1218 / 5413 | 906 / 3273 / 312 / 922 | 0.22 | 0.74 | 0.34 | 0.78 |
| Watchdog: DRIP-ATK only | 1218 / 1305 | 318 / 0 / 900 / 87 | 1.00 | 0.26 | 0.41 | 0.00 |
| Rule: DRIP-ATK only | 1218 / 1305 | 906 / 0 / 312 / 87 | 1.00 | 0.74 | 0.85 | 0.00 |
| Watchdog: SAMESIDE only | 0 / 1416 | 0 / 17 / 0 / 1399 | 0.00 | n/a | 0.00 | 0.01 |
| Rule: SAMESIDE only | 0 / 1416 | 0 / 1068 / 0 / 348 | 0.00 | n/a | 0.00 | 0.75 |
| Watchdog: FLICKER only | 0 / 2692 | 0 / 2 / 0 / 2690 | 0.00 | n/a | 0.00 | 0.00 |
| Rule: FLICKER only | 0 / 2692 | 0 / 2205 / 0 / 487 | 0.00 | n/a | 0.00 | 0.82 |

Per stock, verdict model (context only):

| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |
|---|---|---|---|---|---|---|
| Watchdog: AAPL | 228 / 1108 | 91 / 8 / 137 / 872 | 0.92 | 0.40 | 0.56 | 0.01 |
| Watchdog: MSFT | 239 / 1054 | 19 / 2 / 220 / 813 | 0.90 | 0.08 | 0.15 | 0.00 |
| Watchdog: GOOG | 240 / 1116 | 100 / 1 / 140 / 875 | 0.99 | 0.42 | 0.59 | 0.00 |
| Watchdog: INTC | 256 / 1061 | 15 / 2 / 241 / 803 | 0.88 | 0.06 | 0.11 | 0.00 |
| Watchdog: AMZN | 255 / 1074 | 93 / 6 / 162 / 813 | 0.94 | 0.36 | 0.53 | 0.01 |

Other models on the same orders (context only, not used for any verdict):

| Model | Pooled F1 | DRIP-ATK recall | SAMESIDE FPR | FLICKER FPR |
|---|---|---|---|---|
| causal_g0_1m | 0.409 | 0.261 | 0.012 | 0.001 |
| causal_g0_1m_s2 | 0.549 | 0.438 | 0.117 | 0.009 |
| causal_g0_1m_s3 | 0.586 | 0.462 | 0.100 | 0.000 |
| main | 0.132 | 0.084 | 0.072 | 0.045 |

DRIP-ATK PnL per episode (context only): AAPL -548, MSFT -3,470, GOOG -768, INTC -2,389, AMZN -307
