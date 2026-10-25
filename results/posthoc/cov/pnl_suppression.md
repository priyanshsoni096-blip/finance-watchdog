# PnL suppression

Watchdog `checkpoints/WATCHDOG/cov/model.zip` and the rule (size ≥ 2× depth, cancelled within 1000 events) each act inside the episode. On a flag the exchange cancels the participant's resting large orders and rejects new ones for 200 market events. 50 episodes per source and condition, identical seeds across conditions. 95% CI is a normal approximation over episodes.

Attackers are frozen and do not adapt to surveillance.

| Source | Stock | Kind | PnL: none | PnL: Watchdog | Suppression | PnL: rule | Suppression | Manip. trades/ep none / W / R | Interventions/ep W / R | Blocked spoofs/ep W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | manipulator | 3,186.2 ± 516.9 | -4,557.0 ± 1,210.1 | 243.0% | 140.0 ± 830.5 | 95.6% | 60.2 / 7.6 / 45.1 | 12.1 / 5.5 | 30.0 / 16.1 |
| SPOOFER-04 | INTC | manipulator | 12,624.9 ± 1,162.4 | -132.9 ± 190.2 | 101.1% | 5,655.4 ± 656.7 | 55.2% | 173.1 / 9.0 / 75.9 | 14.9 / 5.9 | 169.5 / 110.1 |
| SCRIPTED-ATK | MSFT | manipulator | 417.6 ± 956.0 | -2,407.6 ± 394.4 | 676.5% | -944.8 ± 371.5 | 326.2% | 29.4 / 5.1 / 15.2 | 8.2 / 4.8 | 5.1 / 5.4 |
| LATEBURST-ATK | MSFT | manipulator | 531.1 ± 674.2 | -2,727.9 ± 429.9 | 613.6% | -973.7 ± 756.6 | 283.3% | 26.5 / 3.2 / 16.0 | 5.0 / 2.6 | 1.7 / 2.1 |
| HONEST | MSFT | legitimate | -961.3 ± 101.3 | -961.3 ± 101.3 | n/a% | -961.3 ± 101.3 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | MSFT | legitimate | -2,480.7 ± 1,025.7 | -4,191.6 ± 1,215.6 | n/a% | -2,045.0 ± 990.2 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 4.7 | 0.1 / 17.6 |
| SCRIPTED-ATK | INTC | manipulator | 748.2 ± 633.6 | -2,457.5 ± 495.8 | 428.5% | -1,280.7 ± 613.4 | 271.2% | 30.0 / 5.1 / 15.2 | 7.8 / 4.7 | 5.0 / 5.2 |
| LATEBURST-ATK | INTC | manipulator | -86.6 ± 1,007.4 | -2,682.2 ± 440.3 | n/a% | -85.8 ± 526.8 | n/a% | 24.8 / 3.2 / 17.3 | 5.4 / 2.7 | 1.5 / 2.0 |
| HONEST | INTC | legitimate | -875.2 ± 78.4 | -875.2 ± 78.4 | n/a% | -875.2 ± 78.4 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | INTC | legitimate | -2,093.5 ± 1,114.0 | -2,139.6 ± 1,110.9 | n/a% | -1,267.5 ± 676.7 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 4.7 | 0.2 / 18.8 |
