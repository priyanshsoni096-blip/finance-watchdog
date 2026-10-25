# Experiment Protocol

What was fixed in advance, what was chosen on which data, and what each number is allowed to support. Read this
before quoting any result; `docs/MODEL_COMPARISON.md` has the numbers, `docs/FINAL_AUDIT.md` has the audit.

---

## 1. Data

| | |
|---|---|
| Source | LOBSTER sample, NASDAQ **2012-06-21**, level 10 |
| Instruments | AAPL, MSFT, GOOG, INTC, AMZN |
| Events | ~2.1 M messages across the five stocks |
| Warm-up | First **30,000 events** of each day discarded (the opening book is unrepresentatively thin: INTC depth was 0.08× its median at event 15,000) |
| Episode | 200 agent steps × 10 market events = **2,000 events** |

**One trading day.** Nothing here supports a claim about regimes, dates or instruments beyond this sample.

---

## 2. Splits

| Split | Sources | Purpose |
|---|---|---|
| `train` | SPOOFER-01..04 on their own stocks; HONEST, FLICKER on AAPL/MSFT/GOOG/INTC | fit parameters |
| `test` | the same sources, different seeds | model and threshold selection |
| `heldout` | SPOOFER-05 (AMZN), SCRIPTED-ATK on all stocks, HONEST/FLICKER on AMZN | never used for selection |
| `drip`, `reversal`, `lateburst` | pre-registered attackers | one scored run each |

**Temporal separation.** The original splits drew episode starts from the whole trading day, so 33 of 50 test
episodes overlapped a training episode (mean 980 of 2,000 events). `--time-split` gives each split a disjoint
stretch: train 0–60%, test 60–80%, held-out 80–100% of the usable day. Measured cost of the original defect:
**0.031 order F1**, below the ±0.058 seed spread. Both protocols are kept; the original remains the default so
published numbers stay reproducible.

**Grouping.** Episodes never cross a split. In the attribution-free experiment, episode ids are unique per
source and the rolling feature window is computed per episode, so no history crosses a boundary
(`tests/test_no_attribution_leakage.py`).

---

## 3. Labels, and when they are available

| Label | Definition | Availability |
|---|---|---|
| **Hindsight** (original) | every step a manipulative order rested; manipulative = `opposite_trades > 0` | retrospective; attaches the outcome to steps *before* the evidence exists |
| **Causal** (CP-7 onward) | from the first step the owner trades opposite a resting order, until removal | available at the step it fires |

The hindsight label is why the original Watchdog failed: it rewarded predicting **who would spoof** over
detecting **the act**. The original Watchdog flagged 99% of SPOOFER-04's orders before any opposite trade, and
13.7% of SPOOFER-03's opposite-trade steps. The causal label keeps 87.0% of train and 87.3% of test hindsight
positives — the change is in *which* steps are positive, not how many.

**Both labels are proxies for intent, not legal determinations.**

---

## 4. The leakage control that matters

The observation contains feature 44 (the participant's trade direction) and feature 45 (the side its resting
orders sit on). The label is defined by exactly that relationship, so **the target is a function of the
input** — definitional leakage, not temporal leakage. A correct temporal split does not fix it.

Measured: a zero-parameter comparison of those two features scores order-level F1 **0.997** in-distribution and
**0.995** on a held-out attacker at FPR 0.000.

**Control:** `watchdog/anonymised_experiment.py` withholds the whole participant block and audits that the
remainder cannot reconstruct the label by another route (no retained feature correlates above 0.381; the best
single-feature rule reaches 0.490 against the withheld pair's 0.948). Guarded by
`tests/test_no_attribution_leakage.py`.

---

## 5. Pre-registered tests

Attacker, legitimate control, hypotheses and verdict model were committed **before any of the test's data
existed**. Each ran once. Git history confirms the ordering.

| Test | Registered in | Verdict model | Hypotheses | Outcome |
|---|---|---|---|---|
| LATEBURST-ATK | pre-CP-7 | `main` | recall vs rule | **went against** the Watchdog (0.24 vs 0.89) |
| LAYER-ATK | pre-CP-7 | `main` | recall vs rule (synthetic market) | **went against** (0.30 vs 1.00) |
| DRIP-ATK | `8688d51` | `causal_g0_1m` | H1 F1 ≥ rule; H2 SAMESIDE FPR ≤ 0.10 | both supported (F1 0.409 vs 0.336; FPR 0.012) |
| REVERSAL-ATK | `ca58bc1` | `cov` | H1 recall ≥ 0.80; H2 FPR ≤ 0.10; H3 F1 ≥ rule | all supported (0.981 / 0.006 / 0.986 vs 0.591) |

**Post-hoc rule.** Once an attacker has been inspected, any other model scored on its data is a description,
not a test. `results/posthoc_rescore.md` is labelled accordingly throughout.

---

## 6. Model and threshold selection

| Decision | Chosen on | Never used |
|---|---|---|
| Watchdog architecture, entropy, γ, training sources | `test` split | held-out, pre-registered data |
| Supervised thresholds (LOGREG, SUP-LSTM, GB) | `train` split only | `test`, held-out |
| Rule parameters (N=2, M=1000 events) | `train` split only | everything else |
| Anonymised-experiment thresholds | `train` split only | `test`, held-out |

One candidate change was evaluated on validation only and **rejected** (percentile impact calibration: it
implied 44–58 spread moves on MSFT/INTC).

---

## 7. Metrics

Order level is primary for attributed detectors (an order is flagged if flagged at any resting step); step and
episode level for the attribution-free experiment.

Reported: precision, recall, F1, FPR, full confusion counts, PR-AUC where class imbalance makes F1 misleading,
and median detection delay.

**Constant-classifier baselines are mandatory.** On a positive-heavy set, "always flag" is a strong F1
baseline: in the attribution-free episode-level evaluation it scores 0.769, beating every learned model. Any
F1 that does not beat the better constant is reported as **not a result**.

**Uncertainty.** 3 training seeds where feasible, mean ± 95% Student-t CI. Seed variance is sometimes larger
than the effect (MSFT Spoofer P&L: +$3,628 to +$15,886), and is reported rather than averaged away. The
non-profitable Spoofers remain single-seed.

---

## 8. What each result class may support

| Claim | Supported by | Not supported by |
|---|---|---|
| RL discovers manipulation unprompted | replay impact model, 3 seeds, impact-off counterfactual | any real-market profitability claim |
| A detector can recover the label | every attributed detector, 0.98–0.99 | any claim that manipulation detection is hard |
| Detection generalises to unseen attacks | CP-7 across four families | the same, without the label-ceiling caveat |
| Detection without attribution | step-level F1 0.392, beating the constant's 0.218 | episode-level F1, which loses to a constant |
| Real-world detection | **nothing** | — |
