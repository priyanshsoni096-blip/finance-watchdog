# Watchdog feature-group ablation (test split)

Watchdog `checkpoints\WATCHDOG\main\model.zip`. Each group is replaced by its training mean; the model is unchanged. Cells show the value after ablation and, in parentheses, the change from no ablation.

| Ablated group | Features | order F1 | order recall | order precision | step recall | step precision | legitimate order FPR |
|---|---|---|---|---|---|---|---|
| none | 0 | 0.772 | 0.797 | 0.748 | 0.865 | 0.953 | 0.043 |
| book price offsets (all levels) | 20 | 0.541 (-0.231) | 0.948 (+0.150) | 0.379 (-0.369) | 0.924 (+0.059) | 0.580 (-0.373) | 0.555 (+0.511) |
| book sizes (all levels) | 20 | 0.821 (+0.049) | 0.764 (-0.033) | 0.886 (+0.138) | 0.805 (-0.060) | 0.989 (+0.036) | 0.006 (-0.037) |
| best level only (4) | 4 | 0.781 (+0.009) | 0.741 (-0.056) | 0.826 (+0.078) | 0.755 (-0.109) | 0.974 (+0.021) | 0.025 (-0.018) |
| all book features (40) | 40 | 0.816 (+0.044) | 0.994 (+0.196) | 0.692 (-0.056) | 0.915 (+0.050) | 0.955 (+0.002) | 0.019 (-0.025) |
| participant: resting size | 1 | 0.792 (+0.020) | 0.761 (-0.036) | 0.826 (+0.078) | 0.820 (-0.044) | 0.972 (+0.019) | 0.018 (-0.026) |
| participant: oldest order age | 1 | 0.770 (-0.001) | 0.796 (-0.001) | 0.747 (-0.002) | 0.862 (-0.003) | 0.951 (-0.002) | 0.047 (+0.004) |
| participant: placed this step | 1 | 0.814 (+0.043) | 0.789 (-0.008) | 0.841 (+0.093) | 0.821 (-0.043) | 0.965 (+0.012) | 0.031 (-0.012) |
| participant: cancelled this step | 1 | 0.773 (+0.001) | 0.795 (-0.002) | 0.752 (+0.004) | 0.861 (-0.003) | 0.953 (+0.000) | 0.040 (-0.004) |
| participant: trade direction | 1 | 0.113 (-0.658) | 0.069 (-0.728) | 0.326 (-0.422) | 0.018 (-0.846) | 0.383 (-0.570) | 0.043 (+0.000) |
| participant: resting side imbalance | 1 | 0.766 (-0.006) | 0.773 (-0.024) | 0.759 (+0.011) | 0.837 (-0.027) | 0.959 (+0.006) | 0.039 (-0.005) |
| all participant features (6) | 6 | 0.001 (-0.771) | 0.001 (-0.797) | 0.025 (-0.723) | 0.000 (-0.864) | 0.007 (-0.946) | 0.008 (-0.035) |
