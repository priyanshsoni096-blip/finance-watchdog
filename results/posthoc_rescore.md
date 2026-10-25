# Post-hoc re-scoring of Watchdog generations (replay market)

**Post-hoc.** Every attacker below has been inspected. Each pre-registered verdict stands as the single run on the model it was registered with (LATEBURST-ATK: `main`, DRIP-ATK: `causal_g0_1m`, REVERSAL-ATK: `cov`). Any other model on that data is a description, not a test.

Generations: `main` = original Watchdog; `causal_g0_1m` = CP-7 round 1 (causal label, γ 0); `cov` = CP-7 round 2 (plus varied-timing training sources). Mean ± 95% Student-t CI over 3 seeds. Rule: size ≥ 2× depth, cancelled within 1000 events.

| Metric | main (n=3) | causal_g0_1m (n=3) | cov (n=3) | Rule |
|---|---|---|---|---|
| in-distribution order F1 | 0.753 ± 0.058 | 0.978 ± 0.015 | 0.984 ± 0.001 | 0.352 |
| legitimate order FPR | 0.031 ± 0.067 | 0.003 ± 0.012 | 0.002 ± 0.002 | 0.820 |
| held-out AMZN legitimate FPR | 0.004 ± 0.018 | 0.001 ± 0.004 | 0.001 ± 0.002 | 0.972 |
| SCRIPTED-ATK train stocks recall | 0.257 ± 0.199 | 0.504 ± 0.024 | 0.991 ± 0.001 | 0.938 |
| SCRIPTED-ATK AMZN recall | 0.052 ± 0.222 | 0.497 ± 0.029 | 0.988 ± 0.011 | 0.950 |
| LATEBURST-ATK recall | 0.217 ± 0.156 | 0.484 ± 0.022 | 0.984 ± 0.001 | 0.885 |
| LATEBURST-ATK legitimate FPR | 0.031 ± 0.067 | 0.003 ± 0.011 | 0.002 ± 0.002 | 0.819 |
| LATEBURST-ATK pooled F1 | 0.329 ± 0.161 | 0.647 ± 0.035 | 0.987 ± 0.003 | 0.462 |
| DRIP-ATK recall | 0.067 ± 0.115 | 0.387 ± 0.273 | 0.989 ± 0.002 | 0.744 |
| DRIP-ATK legitimate FPR | 0.039 ± 0.082 | 0.028 ± 0.053 | 0.012 ± 0.015 | 0.797 |
| DRIP-ATK pooled F1 | 0.106 ± 0.168 | 0.515 ± 0.232 | 0.973 ± 0.025 | 0.336 |
| REVERSAL-ATK recall | 0.139 ± 0.129 | 0.451 ± 0.110 | 0.983 ± 0.005 | 0.964 |
| REVERSAL-ATK legitimate FPR | 0.040 ± 0.087 | 0.007 ± 0.020 | 0.012 ± 0.015 | 0.804 |
| REVERSAL-ATK pooled F1 | 0.226 ± 0.160 | 0.615 ± 0.100 | 0.981 ± 0.010 | 0.591 |
