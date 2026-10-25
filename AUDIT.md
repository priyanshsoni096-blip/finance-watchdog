# Audit Report — Finance Watchdog

Independent audit of correctness, methodology, metrics and test coverage. Everything recorded here was
executed; items not executed are listed as such in §9 rather than assumed to pass.

- **Baseline commit:** `88d7472` (clean tree at audit start)
- **Audit commits:** `4368927` (split defect fix), `83c4bb2` (metric consistency), `08d1272` (simulator invariants)
- **Test suite:** **139 passing**, 18 test files (115 before the audit, +24 added)
- **In flight:** clean-protocol retrain and paired evaluation (§6)

---

## 1. Executive summary

| Area | Status |
|---|---|
| Reported metrics | **Verified.** Every precision, recall, F1 and FPR recomputes from its confusion counts; cross-file values agree |
| Metric discrepancy (0.22 vs 0.24) | **Resolved** — both correct, different experimental conditions (§3) |
| Market simulator accounting | **No defects found** across randomised rollouts on two stocks |
| Train/test separation | **Defect found and fixed** — splits shared market data (§2) |
| Reproducibility | **Verified** — 12/12 scripts reproduce exactly from a clean clone, 2,178 values, 0 differences |
| Model improvement | **Experiment running**, results not yet available |
| Research claims | Conservative; negative results preserved on the main presentation path |

**Overall:** the project's numbers are trustworthy and traceable. One real methodological defect was found,
fixed and regression-tested. The headline in-distribution figures are measured under a protocol now known to
be imperfect, and the corrected measurement is in progress.

---

## 2. Confirmed defect (bug register)

Only one issue met the bar of a *confirmed* defect — reproduced, measured, and affecting reported results.

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
documented in `HANDOFF.md` §6 with the measurements that caught them.

---

## 3. Metric reconciliation

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

`tests/test_metric_consistency.py` recomputes, from source files:

- precision, recall, FPR and F1 from confusion counts in every results file
- `n = tp+fp+fn+tn` and `positives = tp+fn`
- undefined metrics where a denominator is zero (must be NaN, never silently 0)
- multi-seed means against their per-seed values
- the deterministic rule baseline, which must be identical wherever reported
- suppression percentages against the P&L they summarise
- demo episode summaries against their recorded steps

**Result: 10/10 pass.** A further manual audit checked 14 console-displayed values against source files with
no real mismatches.

---

## 4. Market simulator audit

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

## 5. Test report

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
| Synthetic simulator | `test_synthetic_simulator.py` | 6 |
| Episode recording | `test_rollout.py` | 6 |
| **Temporal split (new)** | `test_time_split.py` | **4** |
| Rule detector | `test_baseline_detector.py` | 4 |
| Roster granularity | `test_roster.py` | 3 |
| Multi-seed CIs | `test_multiseed.py` | 3 |
| Layering, ablation, evaluator | 3 files | 6 |
| **Total** | **18 test files** | **139 passing** |

No tests were disabled or weakened. Runtime ~46 s; tests requiring market data skip themselves if absent.

**Not covered by tests:** end-to-end training convergence (too slow for CI), the Streamlit UI (verified
manually instead — all three views, selector, slider and jump buttons exercised in a browser), and the
synthetic market's realism (a modelling question, not a correctness one).

---

## 6. Model improvement — experiment in progress

**Protocol.** Datasets regenerated with disjoint time windows; a Watchdog trained on the clean training split
only; then **both** the new model and the original baseline model evaluated on identical clean data. This
separates the protocol effect from the model effect. Held-out attackers (LATEBURST, LAYER) remain untouched by
model selection.

**Status:** running. Results will be written to `results/temporal_split/` and
`results/temporal_split_baseline_model/`, leaving all baseline artifacts intact.

**Expectation, recorded before seeing results:** in-distribution scores may *fall*, because the previous test
split shared market data with training. A lower clean number would be the more defensible one. Held-out
conclusions should be largely unchanged, since those attackers were never in training.

---

## 7. Reproducibility

Verified from a fresh clone (committed code and checkpoints, raw data copied in): **12/12 scripts reproduce
exactly — 2,178 numeric values compared, 0 differences**, plus all 32 recorded dataset files bit-for-bit and
`configs/calibration.json` identical.

```bash
pip install -r requirements.txt                              # core
pip install -r requirements-dashboard.txt                    # dashboard + figures
python data/download.py                                      # ~600 MB LOBSTER data
python scripts/calibrate.py                                  # impact calibration
python -m pytest tests -q                                    # 139 tests, ~46 s
python evaluation/basic_eval.py --episodes 20                # agents vs controls
python watchdog/dataset.py --episodes 150                    # record surveillance data
python watchdog/evaluate_watchdog.py                         # detector vs rule
streamlit run app.py --server.fileWatcherType none           # console (flag required)
```

The flag is mandatory: the repo package `watchdog/` shadows the PyPI `watchdog` library Streamlit uses.

---

## 8. Research integrity

Preserved and verified during the audit:

- **Negative results stay on the main path.** Claim 2 (generalisation) carries a red "NOT SHOWN" card in the
  console overview; both pre-registered tests went against the detector and are reported that way.
- **Pre-registration is genuine.** Attacker, procedure and hypothesis were committed before any of their data
  existed; each test ran once. Git history confirms the ordering.
- **No held-out tuning.** One candidate design change was evaluated on the validation split only and rejected.
- **Profitability is scoped.** Spoofing pays under the replay impact model and loses in the synthetic market;
  stated as a scope limit, not buried.
- **No real-world claim.** LOBSTER has no participant identities, so detection accuracy on real manipulation
  cannot be established. The console and report say so.
- **The label is a proxy** for intent (trading the opposite side while an order rests), not a legal finding.

---

## 9. Remaining limitations and what is not established

1. **In-distribution metrics are measured under the flawed protocol** until the running experiment completes.
   The corrected figures may be lower.
2. **Generalisation remains unproven.** Recall 0.22–0.30 on unseen attacks against the rule's 0.89–1.00.
3. **Profitability does not transfer** to the synthetic market; which impact model is closer to reality is
   unresolved.
4. **Non-RL baselines not yet compared** (logistic regression, gradient boosting). A supervised diagnostic
   exists but is not a like-for-like detector comparison.
5. **Ablations beyond features** (recurrent vs non-recurrent, reward variants, attack diversity) not run.
6. **Spoofer agents are single-seed** except the two profitable ones (3 seeds each).
7. **One trading day, five stocks**; spoofing is only profitable on the two 1-tick stocks.
8. **Layering cannot be learned**, only scripted, because the replay environment allows one large order per side.
9. **UI is verified manually**, not by automated browser tests.

---

## 10. Verdict

The project is **honest and reproducible**, with one real methodological defect found, fixed and
regression-tested during this audit. Its metrics are traceable to source files and now guarded by automated
consistency checks. Its central negative result — that the learned detector does not generalise to unseen
manipulation — survives scrutiny and is reported prominently rather than hidden.

The main open risk is not a bug but a bound: the in-distribution numbers were measured on partly overlapping
data, and the clean re-measurement is in progress.
