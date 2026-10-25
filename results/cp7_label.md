# Candidate Watchdogs on the test split (CP-7 development)

Test split only; held-out and pre-registered data were not loaded. Order level unless stated. "Flag rate on the act" is the share of steps, on which the owner trades against its own resting manipulative order, that the Watchdog flags.

## Per model

| Metric | main | main_s2 | main_s3 | causal | causal_s2 | causal_s3 | causal_g0 | causal_g0_s2 | causal_g0_s3 | causal_g0_fn2 | causal_g0_fn2_s2 | causal_g0_fn2_s3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| order F1 | 0.772 | 0.761 | 0.727 | 0.839 | 0.835 | 0.676 | 0.846 | 0.960 | 0.962 | 0.845 | 0.742 | 0.974 |
| order precision | 0.748 | 0.765 | 0.949 | 0.960 | 0.933 | 0.958 | 0.959 | 0.948 | 0.940 | 0.960 | 0.706 | 0.959 |
| order recall | 0.797 | 0.758 | 0.589 | 0.745 | 0.755 | 0.522 | 0.756 | 0.973 | 0.986 | 0.754 | 0.783 | 0.989 |
| legitimate order FPR | 0.043 | 0.061 | 0.000 | 0.006 | 0.016 | 0.003 | 0.002 | 0.009 | 0.018 | 0.002 | 0.204 | 0.006 |
| SPOOFER-02_MSFT order recall | 1.000 | 1.000 | 0.941 | 0.971 | 0.996 | 0.971 | 0.996 | 0.994 | 0.996 | 0.994 | 0.996 | 0.996 |
| SPOOFER-02_MSFT flag rate on the act | 0.996 | 0.999 | 0.976 | 0.986 | 1.000 | 0.993 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SPOOFER-03_GOOG order recall | 0.168 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.008 | 0.923 | 0.992 | 0.003 | 0.121 | 0.995 |
| SPOOFER-03_GOOG flag rate on the act | 0.137 | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.008 | 0.932 | 0.997 | 0.001 | 0.124 | 0.997 |
| SPOOFER-04_INTC order recall | 1.000 | 1.000 | 0.660 | 0.991 | 0.993 | 0.485 | 0.996 | 0.996 | 0.987 | 0.994 | 0.996 | 0.993 |
| SPOOFER-04_INTC flag rate on the act | 1.000 | 1.000 | 0.482 | 0.999 | 1.000 | 0.482 | 1.000 | 1.000 | 0.993 | 0.999 | 1.000 | 0.996 |

## Per group (mean ± 95% Student-t CI over seeds)

| Metric | main (n=3) | causal (n=3) | causal_g0 (n=3) | causal_g0_fn2 (n=3) |
|---|---|---|---|---|
| order F1 | 0.753 ± 0.058 | 0.783 ± 0.231 | 0.923 ± 0.166 | 0.854 ± 0.288 |
| order precision | 0.821 ± 0.277 | 0.951 ± 0.037 | 0.949 ± 0.024 | 0.875 ± 0.365 |
| order recall | 0.715 ± 0.274 | 0.674 ± 0.327 | 0.905 ± 0.320 | 0.842 ± 0.319 |
| legitimate order FPR | 0.035 ± 0.078 | 0.008 ± 0.017 | 0.010 ± 0.019 | 0.070 ± 0.287 |
| SPOOFER-02_MSFT order recall | 0.980 ± 0.085 | 0.980 ± 0.035 | 0.995 ± 0.003 | 0.995 ± 0.003 |
| SPOOFER-02_MSFT flag rate on the act | 0.990 ± 0.031 | 0.993 ± 0.017 | 1.000 ± 0.001 | 1.000 ± 0.000 |
| SPOOFER-03_GOOG order recall | 0.056 ± 0.240 | 0.003 ± 0.012 | 0.641 ± 1.364 | 0.373 ± 1.346 |
| SPOOFER-03_GOOG flag rate on the act | 0.046 ± 0.196 | 0.002 ± 0.009 | 0.646 ± 1.376 | 0.374 ± 1.350 |
| SPOOFER-04_INTC order recall | 0.887 ± 0.487 | 0.823 ± 0.727 | 0.993 ± 0.013 | 0.994 ± 0.004 |
| SPOOFER-04_INTC flag rate on the act | 0.827 ± 0.742 | 0.827 ± 0.742 | 0.998 ± 0.009 | 0.998 ± 0.006 |
