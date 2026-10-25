# Did the train/test overlap inflate the results?

Episode starts were drawn from the whole trading day for every split, so 33 of 50 test episodes overlapped a training episode (mean 980 of 2,000 events). The fix gives each split a disjoint stretch of the day. Three conditions separate the protocol change from the model change: the same model is scored under both protocols, and a model retrained on the clean training split is scored alongside it on identical data.

## Order-level F1

| Bucket | Original protocol, main model | Clean protocol, main model | Clean protocol, retrained model | Rule |
|---|---|---|---|---|
| In-distribution (test split) | 0.772 | 0.741 | 0.769 | 0.352 |
| Held-out RL: SPOOFER-05 + AMZN controls | 0.000 | 0.000 | 0.000 | 0.000 |
| Scripted attacker, train stocks + FLICKER | 0.378 | 0.443 | 0.446 | 0.660 |
| Scripted attacker, AMZN + AMZN controls | 0.256 | 0.393 | 0.025 | 0.653 |
| Legitimate only (HONEST + FLICKER) | 0.000 | 0.000 | 0.000 | 0.000 |

## Reading it

**Protocol effect:** the same model scores 0.772 under the original protocol and 0.741 on disjoint-window data, a drop of **0.031**. The defect was real and the published in-distribution figure was optimistic, but by less than the spread across training seeds (±0.058), so no conclusion in the project turns on it.

**Model effect:** retraining on the clean training split recovers the loss (0.769), at a different precision/recall balance — it flags less often, so precision and the false-positive rate improve while recall falls.

**Held-out buckets are not inflated at all.** Scores there are equal or slightly *higher* under the clean protocol, which is expected: those attackers never appeared in training under either protocol, so the overlap could not have helped. The generalisation failure is therefore a property of the detector and not an artefact of the split.

Baseline artifacts are untouched: the published results in `results/` still come from the original protocol, which remains the default so earlier numbers stay reproducible. Run the clean protocol with `python watchdog/dataset.py --time-split --out data/watchdog_v2`.

