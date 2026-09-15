# Simulated attacker order structure

20 episodes per source, fixed seeds, sampled actions for trained Spoofers. Resting times are wall-clock seconds from LOBSTER timestamps. Size ratio is fixed by the environment (spoof = 10× touch depth, lot = ½ touch depth), not chosen by the agents; genuine orders are market orders and always fill.

| Source | Stock | Large orders | Manipulative | Size ÷ lot (fixed) | Cancelled | Filled by market | Resting at episode end | Median resting s (manipulative) | Manipulative resting > 1 s | Opposite-side flips | Opposite trades per manipulative order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | 288 | 70.5% | 20.00 | 95.8% | 1.4% | 2.8% | 0.52 | 43.8% | 21.3% | 4.00 |
| SPOOFER-04 | INTC | 280 | 97.1% | 17.00 | 93.2% | 0.4% | 6.4% | 1.09 | 52.2% | 73.8% | 12.00 |
| SCRIPTED-ATK | MSFT | 200 | 93.5% | 19.00 | 86.5% | 7.5% | 6.0% | 1.34 | 56.1% | 48.3% | 3.00 |
| LATEBURST-ATK | MSFT | 104 | 77.9% | 20.00 | 72.1% | 22.1% | 5.8% | 4.01 | 77.3% | 46.4% | 6.00 |
| SCRIPTED-ATK | INTC | 203 | 94.6% | 17.00 | 88.7% | 5.9% | 5.4% | 2.19 | 63.9% | 46.4% | 3.00 |
| LATEBURST-ATK | INTC | 107 | 80.4% | 17.00 | 75.7% | 11.2% | 13.1% | 4.94 | 75.3% | 46.0% | 6.00 |
