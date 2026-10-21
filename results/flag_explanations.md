# Watchdog decisions explained

Watchdog `checkpoints/WATCHDOG/cov`. Each decision is explained by occlusion: a feature is set to its training mean (0 after normalisation) either at the explained step only or at every earlier step, and the change in the flag probability at the explained step is shown. The counterfactual flips the side of this step's trade. Cases are picked by fixed rules (first match), not by hand. Participant features are as in `evaluation/rollout.py`: resting size and placed / cancelled are log1p multiples of touch depth, age is steps / 20, trade is +1 buy / −1 sell, imbalance is signed (bid minus ask).

## A. True positive: sparse RL Spoofer

`SPOOFER-03_GOOG`, episode 0: large bid (buy) order 23, placed at step 190, removed at step 199 (episode_end), 1 opposite trade(s) while resting, label manipulative. Explained step **197**, flag probability **1.000**.

| Step | From placement | Resting size | Age | Placed | Cancelled | Trade | Imbalance | P(flag) |
|---|---|---|---|---|---|---|---|---|
| 189 | -1 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 190 | +0 | 2.43 | 0.05 | 2.43 | 0.00 |  | +2.43 | 0.000 |
| 191 | +1 | 2.42 | 0.10 | 0.00 | 0.00 |  | +2.42 | 0.000 |
| 192 | +2 | 2.42 | 0.15 | 0.00 | 0.00 |  | +2.42 | 0.000 |
| 193 | +3 | 2.42 | 0.20 | 0.00 | 0.00 |  | +2.42 | 0.000 |
| 194 | +4 | 2.41 | 0.25 | 0.00 | 0.00 |  | +2.41 | 0.000 |
| 195 | +5 | 2.41 | 0.30 | 0.00 | 0.00 |  | +2.41 | 0.000 |
| 196 | +6 | 2.40 | 0.35 | 0.00 | 0.00 |  | +2.40 | 0.000 |
| **197** | +7 | 2.40 | 0.40 | 0.00 | 0.00 | sell | +2.40 | **1.000** |
| 198 | +8 | 2.39 | 0.45 | 0.00 | 0.00 |  | +2.39 | 1.000 |
| 199 | +9 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |

| Occluded | Change in P(flag) |
|---|---|
| counterfactual: trade on the other side | -1.000 |
| trade direction | this step | -1.000 |
| resting side imbalance | this step | -0.001 |
| resting size | this step | -0.000 |
| resting side imbalance | history | -0.000 |
| all 40 book features | this step | +0.000 |
| all 40 book features | history | -0.000 |
| resting size | history | +0.000 |
| cancelled this step | this step | -0.000 |
| cancelled this step | history | -0.000 |
| placed this step | history | +0.000 |
| placed this step | this step | -0.000 |
| trade direction | history | +0.000 |
| oldest order age | history | -0.000 |
| oldest order age | this step | -0.000 |

Largest effects: trade direction | this step (-1.000); resting side imbalance | this step (-0.001); resting size | this step (-0.000). Same trade on the other side: -1.000.

## B. True positive: pre-registered REVERSAL-ATK, the reversed leg

`REVERSAL-ATK_AAPL`, episode 0: large bid (buy) order 1, placed at step 32, removed at step 58 (cancel), 1 opposite trade(s) while resting, label manipulative. Explained step **57**, flag probability **1.000**.

| Step | From placement | Resting size | Age | Placed | Cancelled | Trade | Imbalance | P(flag) |
|---|---|---|---|---|---|---|---|---|
| 31 | -1 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 32 | +0 | 2.40 | 0.05 | 2.40 | 0.00 |  | +2.40 | 0.000 |
| 33 | +1 | 2.40 | 0.10 | 0.00 | 0.00 |  | +2.40 | 0.000 |
| 56 | +24 | 2.41 | 1.25 | 0.00 | 0.00 |  | +2.41 | 0.000 |
| **57** | +25 | 2.41 | 1.30 | 0.00 | 0.00 | sell | +2.41 | **1.000** |
| 58 | +26 | 0.00 | 0.00 | 0.00 | 2.41 |  | +0.00 | 0.000 |
| 59 | +27 | 2.41 | 0.05 | 2.41 | 0.00 |  | -2.41 | 0.000 |
| 60 | +28 | 2.41 | 0.10 | 0.00 | 0.00 |  | -2.41 | 0.000 |

| Occluded | Change in P(flag) |
|---|---|
| counterfactual: trade on the other side | -1.000 |
| trade direction | this step | -1.000 |
| resting size | this step | -0.004 |
| resting side imbalance | history | -0.003 |
| resting side imbalance | this step | -0.002 |
| all 40 book features | history | -0.000 |
| oldest order age | this step | -0.000 |
| all 40 book features | this step | +0.000 |
| resting size | history | +0.000 |
| oldest order age | history | +0.000 |
| cancelled this step | this step | -0.000 |
| placed this step | this step | -0.000 |
| cancelled this step | history | -0.000 |
| placed this step | history | +0.000 |
| trade direction | history | +0.000 |

Largest effects: trade direction | this step (-1.000); resting size | this step (-0.004); resting side imbalance | history (-0.003). Same trade on the other side: -1.000.

## C. False positive: legitimate REVERT (opposite trades just after its own cancel)

`REVERT_AAPL`, episode 10: large ask (sell) order 2, placed at step 54, removed at step 89 (cancel), 0 opposite trade(s) while resting, label legitimate. Explained step **57**, flag probability **0.702**.

| Step | From placement | Resting size | Age | Placed | Cancelled | Trade | Imbalance | P(flag) |
|---|---|---|---|---|---|---|---|---|
| 53 | -1 | 0.00 | 0.00 | 0.00 | 0.00 | buy | +0.00 | 0.000 |
| 54 | +0 | 2.38 | 0.05 | 2.38 | 0.00 |  | -2.38 | 0.000 |
| 55 | +1 | 2.37 | 0.10 | 0.00 | 0.00 |  | -2.37 | 0.004 |
| 56 | +2 | 2.37 | 0.15 | 0.00 | 0.00 |  | -2.37 | 0.184 |
| **57** | +3 | 2.37 | 0.20 | 0.00 | 0.00 |  | -2.37 | **0.702** |
| 58 | +4 | 2.37 | 0.25 | 0.00 | 0.00 |  | -2.37 | 0.216 |
| 88 | +34 | 2.38 | 1.75 | 0.00 | 0.00 |  | -2.38 | 0.058 |
| 89 | +35 | 0.00 | 0.00 | 0.00 | 2.39 |  | +0.00 | 0.000 |
| 90 | +36 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 91 | +37 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |

| Occluded | Change in P(flag) |
|---|---|
| trade direction | history | -0.702 |
| resting side imbalance | history | -0.697 |
| resting size | this step | -0.695 |
| all 40 book features | history | -0.617 |
| cancelled this step | history | -0.555 |
| placed this step | history | +0.270 |
| all 40 book features | this step | +0.264 |
| oldest order age | history | +0.261 |
| resting size | history | +0.243 |
| resting side imbalance | this step | -0.153 |
| cancelled this step | this step | -0.128 |
| placed this step | this step | -0.085 |
| oldest order age | this step | +0.082 |
| trade direction | this step | -0.012 |
| counterfactual: trade on the other side | +0.000 |

Largest effects: trade direction | history (-0.702); resting side imbalance | history (-0.697); resting size | this step (-0.695). Same trade on the other side: +0.000.

## D. True negative: ACCUM trades on the same side as its resting order

`ACCUM_AAPL`, episode 0: large bid (buy) order 0, placed at step 5, removed at step 9 (run_over), 0 opposite trade(s) while resting, label legitimate. Explained step **7**, flag probability **0.000**.

| Step | From placement | Resting size | Age | Placed | Cancelled | Trade | Imbalance | P(flag) |
|---|---|---|---|---|---|---|---|---|
| 4 | -1 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 5 | +0 | 2.42 | 0.05 | 2.42 | 0.00 |  | +2.42 | 0.000 |
| 6 | +1 | 2.42 | 0.10 | 0.00 | 0.00 |  | +2.42 | 0.000 |
| **7** | +2 | 2.42 | 0.15 | 0.00 | 0.00 | buy | +2.42 | **0.000** |
| 8 | +3 | 2.42 | 0.20 | 0.00 | 0.00 |  | +2.42 | 0.000 |
| 9 | +4 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 10 | +5 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |
| 11 | +6 | 0.00 | 0.00 | 0.00 | 0.00 |  | +0.00 | 0.000 |

| Occluded | Change in P(flag) |
|---|---|
| counterfactual: trade on the other side | +1.000 |
| all 40 book features | this step | +0.000 |
| resting side imbalance | this step | +0.000 |
| resting side imbalance | history | +0.000 |
| all 40 book features | history | +0.000 |
| placed this step | history | +0.000 |
| trade direction | this step | +0.000 |
| resting size | this step | -0.000 |
| resting size | history | -0.000 |
| oldest order age | this step | +0.000 |
| cancelled this step | history | -0.000 |
| cancelled this step | this step | -0.000 |
| placed this step | this step | -0.000 |
| oldest order age | history | -0.000 |
| trade direction | history | +0.000 |

Largest effects: all 40 book features | this step (+0.000); resting side imbalance | this step (+0.000); resting side imbalance | history (+0.000). Same trade on the other side: +1.000.

