# Spoof price impact in the synthetic market

300 paired trials (spoofed vs identical control after a 2000-event burn-in). Spoof buy at the best bid, size 10× touch depth. The synthetic spread is 1 tick, so ticks equal spreads.

| Events after placement | Mean mid difference (ticks) | 95% CI | Trials up | Trials down |
|---|---|---|---|---|
| 10 | +0.038 | ±0.025 | 5% | 0% |
| 50 | +0.138 | ±0.068 | 12% | 2% |
| 200 | +0.553 | ±0.152 | 34% | 5% |

Spoof filled by the market within 200 events: median 0.0%, mean 3.6%.

For comparison, the replay environment's calibrated impact of the same spoof (spreads):

| Stock | 10 events | 50 events | 200 events |
|---|---|---|---|
| MSFT | +1.45 | +2.27 | +2.55 |
| INTC | +1.37 | +2.12 | +2.47 |
