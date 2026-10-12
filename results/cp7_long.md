# Candidate Watchdogs on the test split (CP-7 development)

Test split only; held-out and pre-registered data were not loaded. Order level unless stated. "Flag rate on the act" is the share of steps, on which the owner trades against its own resting manipulative order, that the Watchdog flags.

## Per model

| Metric | causal_g0_1m | causal_g0_1m_s2 | causal_g0_1m_s3 |
|---|---|---|---|
| order F1 | 0.982 | 0.971 | 0.982 |
| order precision | 0.973 | 0.953 | 0.974 |
| order recall | 0.990 | 0.990 | 0.990 |
| legitimate order FPR | 0.001 | 0.007 | 0.000 |
| SPOOFER-02_MSFT order recall | 0.994 | 0.994 | 0.994 |
| SPOOFER-02_MSFT flag rate on the act | 1.000 | 1.000 | 1.000 |
| SPOOFER-03_GOOG order recall | 0.997 | 0.997 | 0.997 |
| SPOOFER-03_GOOG flag rate on the act | 1.000 | 1.000 | 1.000 |
| SPOOFER-04_INTC order recall | 0.996 | 0.996 | 0.996 |
| SPOOFER-04_INTC flag rate on the act | 1.000 | 1.000 | 1.000 |

## Per group (mean ± 95% Student-t CI over seeds)

| Metric | causal_g0_1m (n=3) |
|---|---|
| order F1 | 0.978 ± 0.015 |
| order precision | 0.967 ± 0.030 |
| order recall | 0.990 ± 0.000 |
| legitimate order FPR | 0.003 ± 0.009 |
| SPOOFER-02_MSFT order recall | 0.994 ± 0.000 |
| SPOOFER-02_MSFT flag rate on the act | 1.000 ± 0.000 |
| SPOOFER-03_GOOG order recall | 0.997 ± 0.000 |
| SPOOFER-03_GOOG flag rate on the act | 1.000 ± 0.000 |
| SPOOFER-04_INTC order recall | 0.996 ± 0.000 |
| SPOOFER-04_INTC flag rate on the act | 1.000 ± 0.000 |
