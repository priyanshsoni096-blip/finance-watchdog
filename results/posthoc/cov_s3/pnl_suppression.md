# PnL suppression

Watchdog `checkpoints/WATCHDOG/cov_s3/model.zip` and the rule (size ≥ 2× depth, cancelled within 1000 events) each act inside the episode. On a flag the exchange cancels the participant's resting large orders and rejects new ones for 200 market events. 50 episodes per source and condition, identical seeds across conditions. 95% CI is a normal approximation over episodes.

Attackers are frozen and do not adapt to surveillance.

| Source | Stock | Kind | PnL: none | PnL: Watchdog | Suppression | PnL: rule | Suppression | Manip. trades/ep none / W / R | Interventions/ep W / R | Blocked spoofs/ep W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | manipulator | 3,186.2 ± 516.9 | -4,514.1 ± 1,187.7 | 241.7% | 140.0 ± 830.5 | 95.6% | 60.2 / 7.7 / 45.1 | 9.5 / 5.5 | 29.8 / 16.1 |
| SPOOFER-04 | INTC | manipulator | 12,624.9 ± 1,162.4 | -220.4 ± 188.4 | 101.7% | 5,655.4 ± 656.7 | 55.2% | 173.1 / 9.0 / 75.9 | 13.5 / 5.9 | 169.3 / 110.1 |
| SCRIPTED-ATK | MSFT | manipulator | 417.6 ± 956.0 | -2,378.2 ± 392.7 | 669.5% | -944.8 ± 371.5 | 326.2% | 29.4 / 5.1 / 15.2 | 6.9 / 4.8 | 5.1 / 5.4 |
| LATEBURST-ATK | MSFT | manipulator | 531.1 ± 674.2 | -2,721.4 ± 431.0 | 612.4% | -973.7 ± 756.6 | 283.3% | 26.5 / 3.2 / 16.0 | 3.8 / 2.6 | 1.7 / 2.1 |
| HONEST | MSFT | legitimate | -961.3 ± 101.3 | -961.3 ± 101.3 | n/a% | -961.3 ± 101.3 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | MSFT | legitimate | -2,480.7 ± 1,025.7 | -2,480.7 ± 1,025.7 | n/a% | -2,045.0 ± 990.2 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 4.7 | 0.0 / 17.6 |
| SCRIPTED-ATK | INTC | manipulator | 748.2 ± 633.6 | -2,439.3 ± 494.9 | 426.0% | -1,280.7 ± 613.4 | 271.2% | 30.0 / 5.2 / 15.2 | 7.1 / 4.7 | 5.0 / 5.2 |
| LATEBURST-ATK | INTC | manipulator | -86.6 ± 1,007.4 | -2,651.0 ± 438.0 | n/a% | -85.8 ± 526.8 | n/a% | 24.8 / 3.3 / 17.3 | 4.3 / 2.7 | 1.5 / 2.0 |
| HONEST | INTC | legitimate | -875.2 ± 78.4 | -875.2 ± 78.4 | n/a% | -875.2 ± 78.4 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | INTC | legitimate | -2,093.5 ± 1,114.0 | -2,093.5 ± 1,114.0 | n/a% | -1,267.5 ± 676.7 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 4.7 | 0.0 / 18.8 |
