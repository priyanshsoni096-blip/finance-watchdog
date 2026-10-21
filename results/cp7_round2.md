# Candidate Watchdogs on the test split (CP-7 development)

Test split only; held-out and pre-registered data were not loaded. Order level unless stated. Rows for VARIED-SPOOF / ACCUM use the CP-7 round-2 training-only sources' test split. "Flag rate on the act" is the share of steps, on which the owner trades against its own resting manipulative order, that the Watchdog flags.

## Per model

| Metric | causal_g0_1m | causal_g0_1m_s2 | causal_g0_1m_s3 | cov | cov_s2 | cov_s3 |
|---|---|---|---|---|---|---|
| order F1 | 0.982 | 0.971 | 0.982 | 0.984 | 0.984 | 0.985 |
| order precision | 0.973 | 0.953 | 0.974 | 0.974 | 0.972 | 0.974 |
| order recall | 0.990 | 0.990 | 0.990 | 0.995 | 0.995 | 0.995 |
| legitimate order FPR | 0.001 | 0.007 | 0.000 | 0.001 | 0.003 | 0.001 |
| SPOOFER-02_MSFT order recall | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |
| SPOOFER-02_MSFT flag rate on the act | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SPOOFER-03_GOOG order recall | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| SPOOFER-03_GOOG flag rate on the act | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SPOOFER-04_INTC order recall | 0.996 | 0.996 | 0.996 | 0.996 | 0.996 | 0.996 |
| SPOOFER-04_INTC flag rate on the act | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| VARIED-SPOOF order recall | 0.467 | 0.506 | 0.547 | 0.993 | 0.994 | 0.993 |
| ACCUM order FPR | 0.015 | 0.082 | 0.075 | 0.019 | 0.034 | 0.040 |
| VARIED-SPOOF AAPL order recall | 0.491 | 0.527 | 0.518 | 0.991 | 0.991 | 0.991 |
| VARIED-SPOOF GOOG order recall | 0.472 | 0.519 | 0.509 | 0.995 | 1.000 | 0.995 |
| VARIED-SPOOF INTC order recall | 0.436 | 0.470 | 0.564 | 0.992 | 0.992 | 0.992 |
| VARIED-SPOOF MSFT order recall | 0.468 | 0.509 | 0.595 | 0.995 | 0.995 | 0.995 |
| VARIED-SPOOF recall, first act 10+ steps | 0.259 | 0.316 | 0.397 | 1.000 | 1.000 | 1.000 |

## Per group (mean ± 95% Student-t CI over seeds)

| Metric | causal_g0_1m (n=3) | cov (n=3) |
|---|---|---|
| order F1 | 0.978 ± 0.015 | 0.984 ± 0.001 |
| order precision | 0.967 ± 0.030 | 0.973 ± 0.003 |
| order recall | 0.990 ± 0.000 | 0.995 ± 0.001 |
| legitimate order FPR | 0.003 ± 0.009 | 0.002 ± 0.002 |
| SPOOFER-02_MSFT order recall | 0.994 ± 0.000 | 0.994 ± 0.000 |
| SPOOFER-02_MSFT flag rate on the act | 1.000 ± 0.000 | 1.000 ± 0.000 |
| SPOOFER-03_GOOG order recall | 0.997 ± 0.000 | 0.997 ± 0.000 |
| SPOOFER-03_GOOG flag rate on the act | 1.000 ± 0.000 | 1.000 ± 0.000 |
| SPOOFER-04_INTC order recall | 0.996 ± 0.000 | 0.996 ± 0.000 |
| SPOOFER-04_INTC flag rate on the act | 1.000 ± 0.000 | 1.000 ± 0.000 |
| VARIED-SPOOF order recall | 0.506 ± 0.100 | 0.994 ± 0.002 |
| ACCUM order FPR | 0.057 ± 0.092 | 0.031 ± 0.026 |
| VARIED-SPOOF AAPL order recall | 0.512 ± 0.047 | 0.991 ± 0.000 |
| VARIED-SPOOF GOOG order recall | 0.500 ± 0.061 | 0.997 ± 0.007 |
| VARIED-SPOOF INTC order recall | 0.490 ± 0.164 | 0.992 ± 0.000 |
| VARIED-SPOOF MSFT order recall | 0.524 ± 0.160 | 0.995 ± 0.000 |
| VARIED-SPOOF recall, first act 10+ steps | 0.324 ± 0.172 | 1.000 ± 0.000 |
