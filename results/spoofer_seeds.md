# Spoofer results across training seeds

20 episodes per seed, actions sampled from the policy. The no-impact column is the same policy with the price-impact model switched off: profit that survives there does not come from spoofing. Across-seed CIs are Student-t over the seed means.

| Agent | Stock | Seed | PnL per episode | No impact | Trades/ep | Against own spoof/ep | Manipulative orders | Profitable episodes |
|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | main | 3,628.3 ± 1,154.7 | -4,723.5 | 109.8 | 59.5 | 76% | 95% |
| SPOOFER-02 | MSFT | seed11 | 15,885.7 ± 978.7 | -7,795.4 | 167.5 | 166.7 | 96% | 100% |
| SPOOFER-02 | MSFT | seed23 | 14,779.3 ± 2,005.6 | -8,592.8 | 171.2 | 170.5 | 94% | 100% |
| SPOOFER-04 | INTC | main | 13,846.3 ± 2,076.1 | -8,211.1 | 172.8 | 172.2 | 97% | 95% |
| SPOOFER-04 | INTC | seed11 | 14,697.8 ± 2,212.1 | -8,094.0 | 156.8 | 156.1 | 92% | 100% |
| SPOOFER-04 | INTC | seed23 | 14,558.6 ± 2,480.8 | -8,530.6 | 170.2 | 169.4 | 94% | 100% |

| Agent | Stock | Seeds | PnL across seeds | No impact across seeds | Seeds that traded | Seeds profitable |
|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | 3 | 11,431.1 ± 16,843.9 | -7,037.2 ± 5,075.6 | 3/3 | 3/3 |
| SPOOFER-04 | INTC | 3 | 14,367.6 ± 1,134.8 | -8,278.6 ± 561.5 | 3/3 | 3/3 |
