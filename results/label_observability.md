# The manipulation label is directly observable

The label is `opposite_trades > 0`: the order's owner traded on the opposite side while the order rested. Observation feature 44 is the participant's trade direction this step and feature 45 is the side its resting orders sit on, so the label's definition is present in the observation. The signature below is a hand-written comparison of those two features. **Nothing is fitted — there are no parameters and no training.**

## Order-level detection by a zero-parameter signature

| Group | Manipulative / orders | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| train (fitted by nothing) | 4630 / 21716 | 1.000 | 0.990 | 0.995 | 0.000 |
| in-distribution (test split) | 1543 / 7243 | 0.999 | 0.994 | 0.997 | 0.000 |
| held-out scripted attacker | 2420 / 2580 | 1.000 | 0.990 | 0.995 | 0.000 |
| held-out RL attacker (AMZN) | 0 / 2920 | n/a | n/a | n/a | 0.000 |
| legitimate only (HONEST + FLICKER) | 0 / 2688 | n/a | n/a | n/a | 0.000 |

## Step-level

| Group | Positive steps / steps | Precision | Recall | F1 |
|---|---|---|---|---|
| train (fitted by nothing) | 47,072 / 360,000 | 1.000 | 0.780 | 0.877 |
| in-distribution (test split) | 15,831 / 120,000 | 1.000 | 0.781 | 0.877 |
| held-out scripted attacker | 26,985 / 50,000 | 1.000 | 0.259 | 0.411 |
| held-out RL attacker (AMZN) | 0 / 30,000 | n/a | n/a | n/a |
| legitimate only (HONEST + FLICKER) | 0 / 100,000 | n/a | n/a | n/a |

Step-level recall is below 1 because the label is assigned with hindsight to **every** step an order rests, while the signature fires only on the steps where the opposite trade actually occurs. Aggregating to the order ('flagged at any resting step') recovers it.

## Other signatures, for contrast

| Signature | In-distribution order F1 | Held-out scripted order F1 | Legitimate-only FPR |
|---|---|---|---|
| opposite-side signature | 0.997 | 0.995 | 0.000 |
| any trade while resting | 0.995 | 0.995 | 0.000 |
| resting order present | 0.362 | 0.968 | 1.000 |

## What this means

1. **The detection task as posed is close to a tautology.** Given the observation, the label can be read off two features, so a learned detector is not discovering manipulation — at best it is recovering a definition it was handed.
2. **Comparisons against the learned detector must be read in that light.** The rule baseline and the RL Watchdog both score far below this signature, so their gap measures the cost of their own restrictions (the rule may only fire at cancel time; the Watchdog is trained from a sparse reward rather than the label), not the difficulty of the problem.
3. **The simplification is perfect attribution.** The environment tells surveillance which orders and trades belong to the monitored participant. Real surveillance must establish that first, and on anonymised data such as LOBSTER it cannot. That — not pattern recognition — is the hard part of spoofing detection, and this project assumes it away.
4. **The generalisation failure is therefore about the learner, not the features.** A zero-parameter signature transfers to the unseen scripted attacker unchanged, so the Watchdog's collapse on held-out attackers cannot be explained by the observation lacking the signal.

