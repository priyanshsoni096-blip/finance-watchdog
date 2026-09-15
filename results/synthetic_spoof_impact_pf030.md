# Spoof price impact in the synthetic market

300 paired trials (spoofed vs identical control after a 2000-event burn-in). Spoof buy at the best bid, size 10× touch depth. The synthetic spread is 1 tick, so ticks equal spreads.

Parameter overrides: p_follower=0.3. Resulting market: mid_change_share 0.01177, range_2000_ticks 4, touch_depth 7125, exec_share 0.1264.

| Events after placement | Mean mid difference (ticks) | 95% CI | Trials up | Trials down |
|---|---|---|---|---|
| 10 | +0.115 | ±0.054 | 11% | 0% |
| 50 | +0.467 | ±0.137 | 28% | 2% |
| 200 | +1.572 | ±0.260 | 63% | 6% |

Spoof filled by the market within 200 events: median 0.0%, mean 5.4%.

For comparison, the replay environment's calibrated impact of the same spoof (spreads):

| Stock | 10 events | 50 events | 200 events |
|---|---|---|---|
| MSFT | +1.45 | +2.27 | +2.55 |
| INTC | +1.37 | +2.12 | +2.47 |
