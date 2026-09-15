# PnL suppression

Watchdog `checkpoints\WATCHDOG\main\model.zip` and the rule (size ≥ 2× depth, cancelled within 1000 events) each act inside the episode. On a flag the exchange cancels the participant's resting large orders and rejects new ones for 200 market events. 50 episodes per source and condition, identical seeds across conditions. 95% CI is a normal approximation over episodes.

Attackers are frozen and do not adapt to surveillance.

| Source | Stock | Kind | PnL: none | PnL: Watchdog | Suppression | PnL: rule | Suppression | Manip. trades/ep none / W / R | Interventions/ep W / R | Blocked spoofs/ep W / R |
|---|---|---|---|---|---|---|---|---|---|---|
| SPOOFER-02 | MSFT | manipulator | 3,186.2 ± 516.9 | -4,560.5 ± 1,275.1 | 243.1% | 140.0 ± 830.5 | 95.6% | 60.2 / 6.4 / 45.1 | 8.0 / 5.5 | 29.8 / 16.1 |
| SPOOFER-04 | INTC | manipulator | 12,624.9 ± 1,162.4 | 2,165.4 ± 485.0 | 82.8% | 5,655.4 ± 656.7 | 55.2% | 173.1 / 36.4 / 75.9 | 7.9 / 5.9 | 140.0 / 110.1 |
| SCRIPTED-ATK | MSFT | manipulator | 417.6 ± 956.0 | -547.7 ± 594.4 | 231.1% | -944.8 ± 371.5 | 326.2% | 29.4 / 18.1 / 15.2 | 2.6 / 4.8 | 2.3 / 5.4 |
| LATEBURST-ATK | MSFT | manipulator | 531.1 ± 674.2 | -693.4 ± 723.3 | 230.6% | -973.7 ± 756.6 | 283.3% | 26.5 / 19.9 / 16.0 | 1.2 / 2.6 | 0.2 / 2.1 |
| HONEST | MSFT | legitimate | -961.3 ± 101.3 | -961.3 ± 101.3 | n/a% | -961.3 ± 101.3 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | MSFT | legitimate | -2,480.7 ± 1,025.7 | -3,088.5 ± 1,246.8 | n/a% | -2,045.0 ± 990.2 | n/a% | 0.0 / 0.0 / 0.0 | 0.3 / 4.7 | 1.1 / 17.6 |
| SCRIPTED-ATK | INTC | manipulator | 748.2 ± 633.6 | -20.4 ± 525.2 | 102.7% | -1,280.7 ± 613.4 | 271.2% | 30.0 / 21.0 / 15.2 | 2.4 / 4.7 | 1.9 / 5.2 |
| LATEBURST-ATK | INTC | manipulator | -86.6 ± 1,007.4 | 262.0 ± 621.6 | n/a% | -85.8 ± 526.8 | n/a% | 24.8 / 21.3 / 17.3 | 1.3 / 2.7 | 0.3 / 2.0 |
| HONEST | INTC | legitimate | -875.2 ± 78.4 | -875.2 ± 78.4 | n/a% | -875.2 ± 78.4 | n/a% | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| FLICKER | INTC | legitimate | -2,093.5 ± 1,114.0 | -1,882.6 ± 1,103.0 | n/a% | -1,267.5 ± 676.7 | n/a% | 0.0 / 0.0 / 0.0 | 0.4 / 4.7 | 1.6 / 18.8 |
