# Sim-to-sim robustness: synthetic agent-based market

Frozen agents (nothing trained here) in the calibrated synthetic market, where prices come from order matching. 30 episodes per source and condition; identical seeds across conditions. Manipulation gain = reactive PnL − blind-follower PnL (paired, 95% CI).

## Does manipulation still pay?

| Source | Kind | PnL reactive | PnL blind followers | Manipulation gain | Trades against own spoof/ep | Large orders/ep | Manipulative | Filled by market |
|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | manipulator | -2,929.7 ± 685.7 | -2,467.2 ± 661.2 | -462.5 ± 695.9 | 48.6 | 13.6 | 70% | 0% |
| SPOOFER-04 | manipulator | -3,269.8 ± 687.5 | -4,236.3 ± 744.6 | 966.5 ± 944.4 | 122.6 | 34.7 | 67% | 0% |
| SCRIPTED-ATK | manipulator | -16,971.0 ± 3,389.8 | -18,667.7 ± 3,715.1 | 1,696.7 ± 3,827.9 | 8.4 | 2.8 | 99% | 1% |
| LATEBURST-ATK | manipulator | -15,249.0 ± 3,374.3 | -13,979.5 ± 2,803.5 | -1,269.6 ± 3,231.4 | 11.0 | 1.9 | 100% | 0% |
| HONEST | legitimate | -222.7 ± 40.8 | -222.7 ± 40.8 | 0.0 ± 0.0 | 0.0 | 0.0 | n/a% | n/a% |
| FLICKER | legitimate | 117.4 ± 256.1 | -430.2 ± 353.4 | 547.6 ± 472.7 | 0.0 | 10.1 | 0% | 1% |

## Detection in the synthetic market

Watchdog `checkpoints/WATCHDOG/cov/model.zip` (trained on replay data only) vs the rule tuned on the replay train split (size ≥ 2× depth, cancelled within 1000 events), on the same recorded orders.

| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR |
|---|---|---|---|---|---|---|---|---|---|
| Manipulators + legitimate | 1125 / 1893 | 0.96 | 0.99 | 0.98 | 0.06 | 0.59 | 0.96 | 0.74 | 0.96 |
| Legitimate only | 0 / 302 | 0.00 | n/a | 0.00 | 0.08 | 0.00 | n/a | 0.00 | 0.92 |
| SPOOFER-02 + legitimate | 287 / 711 | 0.89 | 0.99 | 0.94 | 0.08 | 0.41 | 0.96 | 0.57 | 0.94 |
| SPOOFER-04 + legitimate | 698 / 1343 | 0.95 | 0.99 | 0.97 | 0.05 | 0.52 | 0.96 | 0.67 | 0.96 |
| SCRIPTED-ATK + legitimate | 83 / 386 | 0.77 | 0.99 | 0.87 | 0.08 | 0.23 | 1.00 | 0.37 | 0.91 |
| LATEBURST-ATK + legitimate | 57 / 359 | 0.70 | 1.00 | 0.83 | 0.08 | 0.17 | 1.00 | 0.29 | 0.92 |

| Source | Positive step rate | Watchdog step flag rate |
|---|---|---|
| SPOOFER-02 | 0.422 | 0.345 |
| SPOOFER-04 | 0.760 | 0.635 |
| SCRIPTED-ATK | 0.157 | 0.085 |
| LATEBURST-ATK | 0.213 | 0.089 |
| HONEST | 0.000 | 0.000 |
| FLICKER | 0.000 | 0.026 |
