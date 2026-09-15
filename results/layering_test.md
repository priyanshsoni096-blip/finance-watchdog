# Pre-registered layering test (synthetic market)

Defined and committed before any of its data or results existed; nothing was changed after seeing them.

**Hypothesis:** Watchdog order-level F1 on LAYER-ATK + FLICKER >= rule F1.

**Result: not supported.** Watchdog F1 = 0.441, rule F1 = 0.733.

Rule tuned on the replay train split: size ≥ 2× depth, cancelled within 1000 events. 30 episodes each of LAYER-ATK (4 layers) and FLICKER in the calibrated synthetic market.

| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR |
|---|---|---|---|---|---|---|---|---|---|
| LAYER-ATK + FLICKER | 400 / 718 | 0.83 | 0.30 | 0.44 | 0.08 | 0.58 | 1.00 | 0.73 | 0.92 |
| LAYER-ATK only | 400 / 404 | 1.00 | 0.30 | 0.46 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| FLICKER only | 0 / 314 | 0.00 | n/a | 0.00 | 0.08 | 0.00 | n/a | 0.00 | 0.93 |

Context (not used for the verdict).

| Layer (0 = first placed) | Manipulative orders | Watchdog recall | Rule recall |
|---|---|---|---|
| 0 | 100 | 0.30 | 1.00 |
| 1 | 100 | 0.30 | 1.00 |
| 2 | 100 | 0.30 | 1.00 |
| 3 | 100 | 0.30 | 1.00 |

LAYER-ATK PnL per episode: reactive -15,000.9 ± 4,362.1, blind followers -14,033.6 ± 2,886.0, manipulation gain -967.3 ± 3,694.0.
