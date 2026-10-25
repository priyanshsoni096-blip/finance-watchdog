# Audit Report — Finance Watchdog

Independent audit of correctness, methodology, metrics and test coverage. Everything recorded here was
executed; items not executed are listed as such in §10 rather than assumed to pass.

- **Baseline commit:** `88d7472` (clean tree at audit start)
- **Test suite:** **145 passing**, 19 test files (115 before the audit, +30 added)
- **Headline finding:** the detection task as posed is close to a tautology (§2). This is the most important
  result in the audit and it limits what the project can claim.
- **In flight:** clean-protocol retrain and paired evaluation (§7)

---

## 1. Executive summary

| Area | Status |
|---|---|
| Reported metrics | **Verified.** Every precision, recall, F1 and FPR recomputes from its confusion counts; cross-file values agree |
| Metric discrepancy (0.22 vs 0.24) | **Resolved** — both correct, different experimental conditions (§4) |
| Market simulator accounting | **No defects found** across randomised rollouts on two stocks |
| Train/test separation | **Defect found and fixed** — splits shared market data (§3) |
| Problem formulation | **Serious limitation found** — the label is readable from two observation features (§2) |
| Reproducibility | **Verified** — 12/12 scripts reproduce exactly from a clean clone, 2,178 values, 0 differences |
| Research claims | Conservative, and now narrower than before this audit |

**Overall:** the code is correct and the numbers are trustworthy and traceable. The problem is not the
implementation — it is the task definition. A zero-parameter signature recovers the manipulation label almost
perfectly, including on held-out attackers, so the learned detector was never solving a hard problem. The
project's honest contribution is the *attacker* side (agents discovering profitable spoofing) plus a
documented negative result on the detector side, not a working surveillance method.

---

## 2. Headline finding: the label is directly observable

**Measured by** `scripts/label_observability.py` → `results/label_observability.md`. Nothing is fitted; there
are no parameters and no training.

The manipulation label is `opposite_trades > 0` — the order's owner traded on the opposite side while the
order rested. The surveillance observation hands the detector two features that describe exactly that event:
feature 44 is the participant's trade direction this step, feature 45 is the side its resting orders sit on.
Comparing their signs therefore *is* the label's definition.

| Group | Manipulative / orders | Precision | Recall | F1 |
|---|---|---|---|---|
| train | 4,630 / 21,716 | 1.000 | 0.990 | 0.995 |
| In-distribution (test split) | 1,543 / 7,243 | 0.999 | 0.994 | **0.997** |
| **Held-out scripted attacker** | 2,420 / 2,580 | 1.000 | 0.990 | **0.995** |
| Legitimate only (HONEST + FLICKER) | 0 / 2,688 | — | — | FPR 0.000 |

For contrast, "a resting order is present" alone gives in-distribution F1 0.362 at FPR 1.000, so it is
specifically the two-feature comparison that carries the label, not the mere presence of a large order.

**Four consequences, all of which narrow the project's claims:**

1. **The detection task is close to a tautology.** A learned detector is not discovering manipulation; at best
   it recovers a definition it was handed in its own input.
2. **Every detector comparison must be read in that light.** The rule baseline (in-distribution order F1 0.35)
   and the RL Watchdog (0.77) both score far below 0.997, so the gap between them measures the cost of their
   own restrictions — the rule may only fire at cancel time, the Watchdog is trained from a sparse reward
   rather than the label — not the difficulty of the problem.
3. **The simplification is perfect attribution.** The environment tells surveillance which orders and trades
   belong to the monitored participant. Real surveillance must establish that first, and on anonymised data
   such as LOBSTER it cannot. That is the hard part of spoofing detection, and this project assumes it away.
4. **The generalisation failure is about the learner, not the features.** A zero-parameter signature transfers
   to the unseen scripted attacker unchanged, so the Watchdog's collapse on held-out attackers cannot be
   explained by the observation lacking the signal. The earlier framing — "learned detectors may not
   generalise to unseen manipulation" — is not supported in this setup and has been corrected.

Locked down by `tests/test_label_observability.py` (6 tests), including a check that the signature reads only
the observation and never the recorded label.

### Supporting evidence: supervised detectors

`scripts/supervised_baseline.py` → `results/supervised_baseline.md`. Both models are fitted on the train split
only, with their decision threshold chosen on train orders only, and scored in the same buckets as the
Watchdog.

| Bucket | Rule | Logistic regression | Gradient boosting | RL Watchdog |
|---|---|---|---|---|
| In-distribution (test split) | 0.35 | 0.58 | **0.96** | 0.77 |
| Scripted attacker, train stocks | 0.66 | 0.44 | **0.99** | 0.38 |
| Scripted attacker, AMZN | 0.65 | 0.21 | **0.99** | 0.26 |

Gradient boosting beats the RL detector everywhere and matches the zero-parameter signature, which is what §2
predicts. This answers the obvious reviewer question — *is reinforcement learning necessary for the detector?*
— with a clear **no, not for this task as posed**.

---

## 3. Confirmed defect (bug register)

One implementation defect was confirmed — reproduced, measured, and affecting reported results.

| Field | Detail |
|---|---|
| **Location** | `env/lob_env.py` (`LimitOrderBookEnv.reset`), `watchdog/dataset.py` (`build_source`) |
| **Severity** | High — affects every in-distribution metric |
| **Root cause** | Episode start indices were drawn from the whole trading day for every split. Splits differed only by random seed, not by the market data they covered. |
| **Reproduction** | Replay the dataset builder's seeding and record `env.t` after each `reset`; compare train and test intervals (episode span 2,000 events) |
| **Measured impact** | **33 of 50 test episodes overlapped a training episode**, mean overlap **980 of 2,000 events**; training episodes covered 38.2% of the day. The test split was unseen *episodes* but not unseen *market data*, so in-distribution scores are likely optimistic. |
| **Fix** | `EnvConfig.start_min` / `start_max` with validation; `watchdog/dataset.py --time-split` giving train 0–60%, test 60–80%, held-out 80–100% of the usable day |
| **Regression tests** | `tests/test_time_split.py` — windows disjoint and ordered, starts inside window, train and held-out episodes cannot overlap, impossible window rejected |
| **Compatibility** | Default behaviour unchanged, so committed baseline results remain reproducible; the clean protocol is opt-in and evaluated separately |

**Not classified as defects.** Reviewed and found correct: Gymnasium space conformance, `reset`/`step` return
contracts, termination vs truncation, recurrent hidden-state resets at episode boundaries, deterministic vs
sampled evaluation (documented and deliberate), checkpoint/config loading, and missing-file handling in the
dashboard. Four historical environment exploits had already been found and fixed before this audit; they are
documented in `HANDOFF.md` with the measurements that caught them.

---

## 4. Metric reconciliation

### The 0.22 vs 0.24 LATEBURST recall question

Both values are correct and describe different experiments. Neither was changed.

| Source | Value | What it is |
|---|---|---|
| `results/lateburst_test.json` | **0.2394** | Single `main` seed — the one pre-registered run |
| `results/multiseed.json` | **0.2173 ± 0.156** | Mean across 3 training seeds |

Verification: the `main` entry inside `multiseed.json` is **0.2394**, identical to the single-seed file;
recall recomputed from raw counts as tp/(tp+fn) matches; the 3-seed mean recomputes from its per-seed values.
**The authoritative value depends on the claim being made** — the pre-registered test result is the single run
(0.239); the robustness statement is the 3-seed mean (0.217 ± 0.156).

**Root cause of the apparent conflict:** presentation, not data. The console showed 3-seed KPIs beside
single-seed charts without labels. Both charts now state their condition.

### Independent recalculation

`tests/test_metric_consistency.py` recomputes, from source files: precision, recall, FPR and F1 from confusion
counts in every results file; `n = tp+fp+fn+tn` and `positives = tp+fn`; undefined metrics where a denominator
is zero (must be NaN, never silently 0); multi-seed means against their per-seed values; the deterministic rule
baseline, which must be identical wherever reported; suppression percentages against the P&L they summarise;
and demo episode summaries against their recorded steps.

**Result: 10/10 pass.** A further manual audit checked 14 console-displayed values against source files with
no real mismatches.

---

## 5. Market simulator audit

`tests/test_env_invariants.py` — 10 tests over randomised rollouts on AAPL and INTC (very different market
structures). **No accounting defects found.**

| Invariant | Result |
|---|---|
| P&L identity: `pnl == cash + inventory × unimpacted mid`, every step | Holds (±1e-6) |
| Every cash/inventory movement explained by a trade, run-over or liquidation | Holds |
| Plain market order moves exactly one lot at the reported fill price | Holds |
| Episodes end flat with no resting orders | Holds |
| Duplicate spoof on one side ignored; cancel idempotent | Holds |
| Refused order (at inventory limit) changes nothing and reports no fill | Holds |
| Reward scale comparable across a $585 and a $27 stock | Holds (<25× spread in reward σ) |
| Observations stay inside the declared Box | Holds |

**Simulated vs not simulated** (unchanged by this audit, restated for clarity): price impact is a fitted model
layered on replayed history — historical prices never react to the agent; run-over uses "price moves through
the level", not queue position; there are no exchange fees or latency; one large order per side is permitted,
so layering cannot occur in the replay environment.

---

## 6. Test report

| Area | File | Tests |
|---|---|---|
| Environment behaviour | `test_env.py` | 25 |
| Price impact & calibration | `test_price_impact.py` | 15 |
| Data loading & alignment | `test_data.py` | 11 |
| Synthetic order book | `test_synthetic_market.py` | 11 |
| Feature normalization | `test_normalization.py` | 10 |
| **Metric consistency (new)** | `test_metric_consistency.py` | **10** |
| **Simulator invariants (new)** | `test_env_invariants.py` | **10** |
| Synthetic env adapter | `test_synthetic_env.py` | 8 |
| Watchdog env & flagging | `test_watchdog_env.py` | 7 |
| **Label observability (new)** | `test_label_observability.py` | **6** |
| Synthetic simulator | `test_synthetic_simulator.py` | 6 |
| Episode recording | `test_rollout.py` | 6 |
| **Temporal split (new)** | `test_time_split.py` | **4** |
| Rule detector | `test_baseline_detector.py` | 4 |
| Roster granularity | `test_roster.py` | 3 |
| Multi-seed CIs | `test_multiseed.py` | 3 |
| Layering, ablation, evaluator | 3 files | 6 |
| **Total** | **19 test files** | **145 passing** |

No tests were disabled or weakened. Runtime ~46 s; tests requiring market data skip themselves if absent.

**Not covered by tests:** end-to-end training convergence (too slow for CI), the Streamlit UI (verified
manually instead — all three views, selector, slider and jump buttons exercised in a browser), and the
synthetic market's realism (a modelling question, not a correctness one).

---

## 7. Model improvement — experiment in progress

**Protocol.** Datasets regenerated with disjoint time windows; a Watchdog trained on the clean training split
only; then **both** the new model and the original baseline model evaluated on identical clean data. This
separates the protocol effect from the model effect. Held-out attackers remain untouched by model selection.

**Status:** running. Results will be written to `results/temporal_split/` and
`results/temporal_split_baseline_model/`, leaving all baseline artifacts intact.

**Expectation, recorded before seeing results:** in-distribution scores may *fall*, because the previous test
split shared market data with training. A lower clean number would be the more defensible one. In light of §2,
this experiment now settles a narrower question than it was designed for: whichever way it lands, the detector
is being compared against a 0.997 zero-parameter signature.

---

## 8. Reproducibility

Verified from a fresh clone (committed code and checkpoints, raw data copied in): **12/12 scripts reproduce
exactly — 2,178 numeric values compared, 0 differences**, plus all 32 recorded dataset files bit-for-bit and
`configs/calibration.json` identical.

```bash
pip install -r requirements.txt                              # core
pip install -r requirements-dashboard.txt                    # dashboard + figures
python data/download.py                                      # ~600 MB LOBSTER data
python scripts/calibrate.py                                  # impact calibration
python -m pytest tests -q                                    # 145 tests, ~46 s
python evaluation/basic_eval.py --episodes 20                # agents vs controls
python watchdog/dataset.py --episodes 150                    # record surveillance data
python watchdog/evaluate_watchdog.py                         # detector vs rule
python scripts/label_observability.py                        # the headline finding (section 2)
python scripts/supervised_baseline.py                        # is RL necessary? (section 2)
streamlit run app.py --server.fileWatcherType none           # console (flag required)
```

The flag is mandatory: the repo package `watchdog/` shadows the PyPI `watchdog` library Streamlit uses.

---

## 9. Research integrity

Preserved and verified during the audit:

- **Negative results stay on the main path,** including the one this audit produced. §2 is reported as the
  headline finding rather than a footnote, and it is the finding least favourable to the project.
- **Pre-registration is genuine.** Attacker, procedure and hypothesis were committed before any of their data
  existed; each test ran once. Git history confirms the ordering.
- **No held-out tuning.** One candidate design change was evaluated on the validation split only and rejected.
  The supervised baselines' thresholds were chosen on train orders only.
- **Profitability is scoped.** Spoofing pays under the replay impact model and loses in the synthetic market;
  stated as a scope limit, not buried.
- **No real-world claim.** LOBSTER has no participant identities, so detection accuracy on real manipulation
  cannot be established. §2 sharpens this: the missing ingredient is attribution, not pattern recognition.
- **The label is a proxy** for intent, not a legal finding — and, per §2, a proxy the observation reveals.

---

## 10. Remaining limitations and what is not established

1. **The detection task is near-tautological** (§2). No claim that the system "detects spoofing" in a
   meaningful sense is supported; what is supported is that the label is recoverable, and that the RL
   formulation recovers it worse than a two-feature comparison does.
2. **Reinforcement learning is not justified for the detector.** Gradient boosting dominates it on every
   bucket measured.
3. **In-distribution metrics are measured under the flawed protocol** until the running experiment completes
   (§7). The corrected figures may be lower.
4. **Profitability does not transfer** to the synthetic market; which impact model is closer to reality is
   unresolved.
5. **Ablations beyond features** (recurrent vs non-recurrent, reward variants, attack diversity) not run.
6. **Spoofer agents are single-seed** except the two profitable ones (3 seeds each).
7. **One trading day, five stocks**; spoofing is only profitable on the two 1-tick stocks.
8. **Layering cannot be learned**, only scripted, because the replay environment allows one large order per side.
9. **UI is verified manually**, not by automated browser tests.

---

## 11. Verdict

The implementation is **correct, reproducible and honestly reported**. One real defect was found, fixed and
regression-tested. The metrics are traceable to source files and now guarded by automated consistency checks.

The audit's substantive result is about the research design rather than the code: **the detector half of the
project does not pose a hard problem.** The manipulation label can be read off two of the detector's own input
features, a zero-parameter signature scores 0.997 in-distribution and 0.995 on a held-out attacker, and
gradient boosting beats the RL detector everywhere. The learned detector's poor generalisation is therefore a
property of the learner and the reward, not evidence about manipulation detection.

What survives intact: the adversarial *attacker* side — RL agents independently discovering a spoofing
strategy that is profitable under a calibrated impact model and structurally similar to documented cases — the
simulator, the calibration, the reproduction pipeline, and a clearly documented negative result. Those are
worth presenting. A claim to have built a working spoofing detector is not.
