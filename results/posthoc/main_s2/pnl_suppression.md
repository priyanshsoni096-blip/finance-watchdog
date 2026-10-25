# PnL suppression

Watchdog `checkpoints/WATCHDOG/main_s2/model.zip` and the rule (size ≥ 2× depth, cancelled within 1000 events) each act inside the episode. On a flag the exchange cancels the participant's resting large orders and rejects new ones for 200 market events. 50 episodes per source and condition, identical seeds across conditions. 95% CI is a normal approximation over episodes.

Attackers are frozen and do not adapt to surveillance.

| Source | Stock | Kind | PnL: none | PnL: Watchdog | Suppression | PnL: rule | Suppression | Manip. trades/ep none / W / R | Interventions/ep W / R | Blocked spoofs/ep W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | manipulator | 3,186.2 ± 516.9 | -4,856.6 ± 1,299.4 | 252.4% | 140.0 ± 830.5 | 95.6% | 60.2 / 2.3 / 45.1 | 8.3 / 5.5 | 32.3 / 16.1 |
| SPOOFER-04 | INTC | manipulator | 12,624.9 ± 1,162.4 | 457.9 ± 295.6 | 96.4% | 5,655.4 ± 656.7 | 55.2% | 173.1 / 18.9 / 75.9 | 8.9 / 5.9 | 158.0 / 110.1 |
| SCRIPTED-ATK | MSFT | manipulator | 417.6 ± 956.0 | -1,495.9 ± 541.9 | 458.2% | -944.8 ± 371.5 | 326.2% | 29.4 / 11.3 / 15.2 | 4.5 / 4.8 | 3.0 / 5.4 |
| LATEBURST-ATK | MSFT | manipulator | 531.1 ± 674.2 | -1,611.3 ± 512.8 | 403.4% | -973.7 ± 756.6 | 283.3% | 26.5 / 10.8 / 16.0 | 3.2 / 2.6 | 0.5 / 2.1 |
| HONEST | MSFT | legitimate | -961.3 ± 101.3 | -961.3 ± 101.3 | n/a% | -961.3 ± 101.3 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | MSFT | legitimate | -2,480.7 ± 1,025.7 | -2,875.8 ± 1,545.9 | n/a% | -2,045.0 ± 990.2 | n/a% | 0.0 / 0.0 / 0.0 | 1.3 / 4.7 | 5.1 / 17.6 |
| SCRIPTED-ATK | INTC | manipulator | 748.2 ± 633.6 | -1,232.8 ± 438.4 | 264.8% | -1,280.7 ± 613.4 | 271.2% | 30.0 / 12.6 / 15.2 | 4.1 / 4.7 | 3.0 / 5.2 |
| LATEBURST-ATK | INTC | manipulator | -86.6 ± 1,007.4 | -1,761.6 ± 748.4 | n/a% | -85.8 ± 526.8 | n/a% | 24.8 / 12.1 / 17.3 | 2.9 / 2.7 | 0.5 / 2.0 |
| HONEST | INTC | legitimate | -875.2 ± 78.4 | -875.2 ± 78.4 | n/a% | -875.2 ± 78.4 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | INTC | legitimate | -2,093.5 ± 1,114.0 | -2,296.3 ± 1,184.1 | n/a% | -1,267.5 ± 676.7 | n/a% | 0.0 / 0.0 / 0.0 | 1.0 / 4.7 | 4.3 / 18.8 |
