# Synthetic market, layering and PnL suppression across Watchdog seeds

**Post-hoc** for every model other than `main` (all attackers here were inspected earlier). `main` = original Watchdog, `cov` = CP-7 round 2. Mean ± 95% Student-t CI over seeds; n is the number of seeds with that measurement (synthetic and layering were run for `main` once, so it has no CI there). In PnL rows the rule column is the rule acting as surveillance.

PnL without surveillance (same seeds, 50 episodes): SPOOFER-02 MSFT +3,186 ± 517, SPOOFER-04 INTC +12,625 ± 1,162.

| Metric | main | cov | Rule |
|---|---|---|---|
| synthetic: all manipulators precision | 0.876 (n=1) | 0.976 ± 0.032 (n=3) | 0.595 |
| synthetic: all manipulators recall | 0.483 (n=1) | 0.985 ± 0.013 (n=3) | 0.964 |
| synthetic: all manipulators f1 | 0.622 (n=1) | 0.980 ± 0.011 (n=3) | 0.736 |
| synthetic: legitimate FPR | 0.113 (n=1) | 0.030 ± 0.108 (n=3) | 0.917 |
| synthetic: SPOOFER-02 F1 | 0.719 (n=1) | 0.958 ± 0.048 (n=3) | 0.575 |
| synthetic: SPOOFER-04 F1 | 0.550 (n=1) | 0.981 ± 0.020 (n=3) | 0.674 |
| synthetic: SCRIPTED-ATK F1 | 0.561 (n=1) | 0.944 ± 0.165 (n=3) | 0.375 |
| synthetic: LATEBURST-ATK F1 | 0.345 (n=1) | 0.933 ± 0.233 (n=3) | 0.292 |
| LAYER-ATK recall | 0.300 (n=1) | 0.990 ± 0.025 (n=3) | 1.000 |
| LAYER-ATK pooled F1 | 0.441 (n=1) | 0.984 ± 0.035 (n=3) | 0.733 |
| LAYER-ATK FLICKER FPR | 0.076 (n=1) | 0.029 ± 0.090 (n=3) | 0.927 |
| PnL suppression: SPOOFER-02 MSFT (%) | 243.248 ± 22.663 (n=3) | 241.909 ± 2.531 (n=3) | 95.606 |
| PnL with Watchdog: SPOOFER-02 MSFT ($) | -4,564.152 ± 722.079 (n=3) | -4,521.489 ± 80.640 (n=3) | 139.992 |
| PnL suppression: SPOOFER-04 INTC (%) | 87.519 ± 19.059 (n=3) | 101.411 ± 0.862 (n=3) | 55.205 |
| PnL with Watchdog: SPOOFER-04 INTC ($) | 1,575.714 ± 2,406.181 (n=3) | -178.098 ± 108.819 (n=3) | 5,655.359 |
| interventions/episode on legitimate: FLICKER MSFT | 0.553 ± 1.739 (n=3) | 0.013 ± 0.057 (n=3) | 4.720 |
| interventions/episode on legitimate: FLICKER INTC | 0.487 ± 1.300 (n=3) | 0.007 ± 0.029 (n=3) | 4.740 |
