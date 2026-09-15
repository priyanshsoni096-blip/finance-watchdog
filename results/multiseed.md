# Watchdog across training seeds

Seeds (checkpoint tags): main, main_s2, main_s3. 95% CI uses Student-t with n = 3. Every seed is scored on the same recorded data. The rule baseline does not depend on the seed.

| Metric | main | main_s2 | main_s3 | Mean ± 95% CI | Rule |
|---|---|---|---|---|---|
| in-distribution order F1 | 0.772 | 0.761 | 0.727 | 0.753 ± 0.058 | 0.352 |
| in-distribution order precision | 0.748 | 0.765 | 0.949 | 0.821 ± 0.277 | 0.217 |
| in-distribution order recall | 0.797 | 0.758 | 0.589 | 0.715 ± 0.274 | 0.941 |
| in-distribution step precision | 0.953 | 0.952 | 0.994 | 0.966 ± 0.060 | — |
| in-distribution step recall | 0.865 | 0.865 | 0.501 | 0.743 ± 0.522 | — |
| legitimate-only order FPR | 0.045 | 0.049 | 0.000 | 0.031 ± 0.067 | 0.823 |
| held-out AMZN legitimate order FPR | 0.013 | 0.000 | 0.000 | 0.004 ± 0.018 | 0.972 |
| LATEBURST-ATK order recall | 0.239 | 0.266 | 0.147 | 0.217 ± 0.156 | 0.885 |
| LATEBURST-ATK order precision | 0.659 | 0.663 | 1.000 | 0.774 ± 0.486 | 0.312 |
| LATEBURST-ATK order FPR | 0.048 | 0.053 | 0.000 | 0.034 ± 0.072 | 0.757 |
