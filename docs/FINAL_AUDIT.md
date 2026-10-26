# Audit Report — Finance Watchdog

Independent audit of correctness, methodology, metrics, test coverage and presentation, covering the **merged**
project: `main` (protocol audit, dashboard, documentation) and `origin/arnav` (the CP-7 detector research),
reconciled in commit `5a32de9`. Everything recorded here was executed. Work that was *not* executed is named as
such in §G and §J rather than assumed to pass.

- **Merge base:** `d9e242b` · **main before merge:** `4c7201c` (tag `pre-merge-main`) · **merged:** `5a32de9`
- **Test suite:** **191 passing**, 24 test files
- **Headline finding:** the detection task is close to a tautology (§C). CP-7 reached the label's ceiling; it did
  not make the problem hard.
- **Independent verification:** Arnav's CP-7 `cov` result reproduced exactly on a second machine (§D)

---

## A. Executive audit report

| Area | Status |
|---|---|
| Branch reconciliation | **Done.** 19 + 17 divergent commits merged; three conflicts resolved keeping both sides |
| Reported metrics | **Verified.** Every precision, recall, F1 and FPR recomputes from its confusion counts |
| CP-7 headline result | **Independently reproduced** — 0.984 / 0.001 / 0.991 / 0.983, matching to 3 d.p. (§D) |
| Market simulator accounting | **No defects found** across randomised rollouts on two stocks |
| Train/test separation | **Defect found and fixed**; measured cost 0.031 F1, below seed noise (§B) |
| Problem formulation | **Serious limitation, confirmed twice independently** (§C) |
| Is RL necessary for detection? | **No** — a supervised LSTM and a zero-parameter rule both match it (§D) |
| Reproducibility | **Verified** — 12/12 scripts, 2,178 values, 0 differences, from a clean clone |

**Overall.** The implementation is correct and the numbers are trustworthy. The research arc is now complete and
coherent: an RL detector failed on unseen attacks; the failure was correctly diagnosed as a labelling artefact;
the fix worked and was confirmed on four held-out attack families. But the same audit shows that the ceiling it
reached is occupied by a rule with no parameters, so the detector result is a demonstration of **recoverability**,
not of RL's value. The defensible contributions are the attacker side, the diagnostic methodology, and an
unusually honest negative result.

**Remaining blockers for a real-world claim:** none of them are code. They are the attribution assumption (§C),
single-day data, and the absence of any labelled real spoofing.

---

## B. Bug register

Confirmed defects only — reproduced, measured, and affecting reported results.

### B1. Splits shared market data

| Field | Detail |
|---|---|
| **Location** | `env/lob_env.py` (`LimitOrderBookEnv.reset`), `watchdog/dataset.py` (`build_source`) |
| **Severity** | High — affected every in-distribution metric |
| **Root cause** | Episode starts were drawn from the whole trading day for every split; splits differed only by seed |
| **Reproduction** | Replay the builder's seeding, record `env.t` per `reset`, compare train and test intervals |
| **Measured impact** | **33 of 50 test episodes overlapped a training episode**, mean 980 of 2,000 events; training covered 38.2% of the day |
| **Fix** | `EnvConfig.start_min` / `start_max` with validation; `watchdog/dataset.py --time-split` |
| **Measured cost** | Same model: 0.772 → **0.741** in-distribution F1, a drop of **0.031**, below the ±0.058 seed spread. Held-out buckets were not inflated at all. |
| **Regression test** | `tests/test_time_split.py` (4 tests) |

### B2. Hindsight labels trained the wrong behaviour

| Field | Detail |
|---|---|
| **Location** | `watchdog/dataset.py` — label construction |
| **Severity** | High — caused the original Watchdog's held-out failure |
| **Root cause** | The label marked **every** step an order rested, including steps before any opposite trade, so the objective rewarded predicting *who would spoof* over detecting *the act* |
| **Reproduction** | `scripts/flag_timing_diagnostic.py` — the original Watchdog flagged **99%** of SPOOFER-04's orders before any opposite trade, and only **13.7%** of SPOOFER-03's opposite-trade steps |
| **Fix** | `causal_labels()`: positives begin at the first step the owner trades opposite a resting manipulative order. Every causal positive is also a hindsight positive. |
| **Measured effect** | Held-out recall 0.22–0.30 → **0.98–0.99** across four attack families |
| **Measured label gap** | The causal label keeps 87.0% of train and 87.3% of test hindsight positives, so the change is not mainly in volume but in *which* steps are positive: it removes exactly the pre-act steps the detector had been keying on |
| **Regression test** | `tests/test_causal_labels.py` (7 tests), including the docstring's claim that every causal positive is also a hindsight positive |
| **Found by** | Arnav |

**Reviewed and found correct** (not defects): Gymnasium space conformance, `reset`/`step` contracts, termination
vs truncation, recurrent hidden-state resets at episode boundaries, deterministic vs sampled evaluation
(documented and deliberate), checkpoint and config loading, missing-file handling in the dashboard, and
normalisation statistics fitted on training data only. Four earlier environment exploits were already found and
fixed before this audit, each caught by decomposing P&L rather than reading headline profit.

---

## C. Research-design report — the label is directly observable

**The single most important finding, reached independently by both work streams.**

The label is `opposite_trades > 0`: the order's owner traded the opposite side while it rested. The surveillance
observation supplies feature 44 (the participant's trade direction this step) and feature 45 (the side its
resting orders sit on). Comparing their signs *is* the label's definition.

Measured with **no parameters and no fitting** (`scripts/label_observability.py`):

| Group | Manipulative / orders | Precision | Recall | Order F1 |
|---|---|---|---|---|
| In-distribution (test split) | 1,543 / 7,243 | 0.999 | 0.994 | **0.997** |
| Held-out scripted attacker | 2,420 / 2,580 | 1.000 | 0.990 | **0.995** |
| Legitimate only | 0 / 2,688 | — | — | FPR **0.000** |

Arnav reached the same conclusion from the other direction, building **LABEL-RULE** — the label's definition
written as a rule — and stating plainly that it is *"the ceiling, perfect by construction, not a competitor."*
It scores 0.983–0.995 recall at 0.000 FPR on every held-out attack family (`results/baselines_final.md`).

Two independent confirmations that this is not an artefact:
- **Not caused by the split overlap.** Re-measured on clean temporal-split data, the signature scores 0.995
  instead of 0.997.
- **Not caused by order presence.** "A resting order is present" alone scores F1 0.362 at FPR 1.000, so it is
  specifically the two-feature comparison that carries the label.

### What this means

1. **Information available at decision time.** The two features are genuinely available to the detector at the
   step it must decide, so this is not temporal leakage. It is *definitional* leakage: the target is a function
   of the input.
2. **It requires privileged attribution.** The environment tells surveillance which orders and trades belong to
   the monitored participant. A regulator or exchange with account-level data has this; an anonymised feed such
   as LOBSTER does not. **Establishing attribution is the hard part of real spoofing surveillance, and this
   project assumes it away.**
3. **What the evaluation measures.** Recovery of the label's definition, not detection of manipulation.
4. **The target is a simulator-specific proxy** for intent, not a legal determination.

### The leakage-resistant formulation — designed, not run

The defensible next experiment is to **withhold per-participant attribution** and require detection from
aggregate order flow (the 40 book features already include the participant's resting size). The target would
have to become episode- or window-level ("manipulation is occurring"), because without attribution there is no
"this participant's order" to label.

**This was not run.** It requires rebuilding the datasets and retraining, and doing it properly is a research
project rather than a finishing task. It is recorded as the first item of future work. The existing feature
ablation is a partial indication: removing the participant features entirely leaves order-level F1 at 0.001.

---

## D. Model comparison report

### D0. Cross-platform reproduction of a pre-registered test

Arnav's `data/watchdog/drip` and `reversal` datasets are gitignored, so both were **regenerated from scratch on
this machine** (Windows/x86) and their pre-registered tests re-run end to end, against the result files he
produced on macOS. Comparison against a pre-regeneration backup of the committed files:

| Test | Numeric values | Bit-identical | Float noise (<1e-10 rel.) | Real differences |
|---|---|---|---|---|
| DRIP-ATK | 1,164 | 1,158 | 5 | **0** |
| REVERSAL-ATK | 1,615 | 1,609 | 5 | **0** |

The only non-identical values are the five per-stock attacker P&L figures, differing at relative ~1e-15 —
floating-point accumulation order across platforms — plus wall-clock runtime. **Every detection metric,
confusion count and hypothesis verdict is bit-identical**: DRIP H1 supported (Watchdog F1 0.409 vs rule 0.336)
and H2 supported (SAMESIDE FPR 0.012); REVERSAL H1 supported (recall 0.981), H2 supported (legitimate FPR
0.006), H3 supported (F1 0.986 vs rule 0.591). Recording, scoring and verdict are reproducible across operating
systems from the committed code and checkpoints alone.

### D1. Independent verification of CP-7

Arnav's results were produced on macOS. I re-ran `checkpoints/WATCHDOG/cov` on Windows from his committed
checkpoint:

| Metric | My re-run | Arnav's committed | Match |
|---|---|---|---|
| In-distribution order F1 | 0.984 | 0.984 | ✓ |
| Legitimate order FPR | 0.001 | 0.001 | ✓ |
| SCRIPTED-ATK train-stocks recall | 0.991 | 0.991 | ✓ |
| SCRIPTED-ATK AMZN recall | 0.983 | 0.983 | ✓ |

### D2. Three Watchdog generations (post-hoc, 3 seeds, mean ± 95% CI)

Source `results/posthoc_rescore.md`. Only LATEBURST (for `main`) and REVERSAL (for `cov`) were pre-registered
verdicts; the rest describe models on data since inspected and decide nothing.

| Metric | `main` | `causal_g0_1m` | `cov` (CP-7 r2) | Rule |
|---|---|---|---|---|
| In-distribution order F1 | 0.753 ± 0.058 | 0.978 ± 0.015 | **0.984 ± 0.001** | 0.352 |
| Legitimate order FPR | 0.031 ± 0.067 | 0.003 ± 0.012 | **0.002 ± 0.002** | 0.820 |
| LATEBURST-ATK recall | 0.217 ± 0.156 | 0.484 ± 0.022 | **0.984 ± 0.001** | 0.885 |
| DRIP-ATK recall | 0.067 ± 0.115 | 0.387 ± 0.273 | **0.989 ± 0.002** | 0.744 |
| REVERSAL-ATK recall | 0.139 ± 0.129 | 0.451 ± 0.110 | **0.983 ± 0.005** | 0.964 |

### D3. Is RL necessary? No.

All detectors fed the same 46-dim observation, same causal labels, thresholds chosen on training data only
(`results/baselines_final.md`; recall / legitimate FPR):

| Dataset | Label ceiling | SUP-LSTM | RL `cov` | LOGREG | Size+cancel rule |
|---|---|---|---|---|---|
| Original test split | 0.995 / 0.000 | 0.995 / 0.000 | 0.995 / 0.002 | 0.469 / 0.123 | 0.941 / 0.812 |
| Held-out (SCRIPTED, AMZN) | 0.990 / 0.000 | 0.990 / 0.000 | 0.990 / 0.004 | 0.198 / 0.000 | 0.940 / 0.869 |
| LATEBURST-ATK | 0.983 / 0.000 | 0.983 / 0.000 | 0.984 / 0.002 | 0.218 / 0.098 | 0.885 / 0.819 |
| DRIP-ATK | 0.989 / 0.000 | 0.989 / 0.000 | 0.989 / 0.012 | 0.324 / 0.144 | 0.744 / 0.797 |
| REVERSAL-ATK | 0.984 / 0.000 | 0.984 / 0.001 | 0.983 / 0.012 | 0.218 / 0.099 | 0.964 / 0.804 |

My own supervised comparison on the original hindsight labels agrees from a different angle: gradient boosting
reaches order F1 0.96 in-distribution and 0.99 on both scripted buckets, against the original Watchdog's 0.77 /
0.38 / 0.26 (`results/supervised_baseline.md`).

**Verdict.** A supervised LSTM matches the RL Watchdog everywhere, with marginally fewer false positives, and
trains in minutes. **Gradient boosting or a supervised sequence model is the appropriate detector for this task
as posed.** Logistic regression fails (0.20–0.47 recall), which is informative: the task needs the *interaction*
between trade side and resting side — but that is a two-feature interaction, which is exactly §C's point. With
γ = 0, the CP-7 Watchdog's RL training is cost-weighted classification by policy gradient; the sequential
machinery is doing no work. RL would matter where feedback is delayed or partial, which is not tested here.

**Where RL is genuinely justified in this project: the attacker side**, where there is no label to supervise
against and the agent must discover a strategy from profit alone.

---

## E. Generalization report

| Attack family | Nature | Original Watchdog | CP-7 `cov` | Rule | Label ceiling |
|---|---|---|---|---|---|
| SCRIPTED-ATK (train stocks) | unseen structure | 0.257 ± 0.199 | 0.991 ± 0.001 | 0.938 | ~0.99 |
| SCRIPTED-ATK (AMZN) | unseen structure + stock | 0.052 ± 0.222 | 0.988 ± 0.011 | 0.950 | 0.990 |
| LATEBURST-ATK | sparse, late trading | 0.217 ± 0.156 | 0.984 ± 0.001 | 0.885 | 0.983 |
| DRIP-ATK | slow drip of trades | 0.067 ± 0.115 | 0.989 ± 0.002 | 0.744 | 0.989 |
| REVERSAL-ATK | cancel, then reverse | 0.139 ± 0.129 | 0.983 ± 0.005 | 0.964 | 0.984 |
| LAYER-ATK (synthetic market) | layered orders | 0.30 | **0.990 ± 0.025** | 1.00 | — |
| Held-out RL (SPOOFER-05, AMZN) | no manipulation present | n/a | n/a | n/a | n/a |

**The common factor in every original-Watchdog failure was act timing**, which is exactly what the hindsight
label obscured (§B2). Once the label marked the act rather than the actor, every family was recovered.

**Pre-registration held.** Attacker, legitimate control and hypotheses were committed before any of their data
existed, each test run once, verdict model frozen in advance. Git history confirms the ordering. The held-out RL
bucket (SPOOFER-05, AMZN) contains no manipulation, because calibrated impact makes spoofing unprofitable on
wide-spread stocks — a real weakness in the evaluation design, not a result.

---

## F. Attacker and simulator report

**Attacker.** Spoofing emerged without instruction on the two 1-tick stocks: INTC +$14,368 ± 1,135 and MSFT
+$11,431 ± 16,844 per episode across 3 seeds, 3/3 seeds on both. With price impact switched off the same trading
loses $7–8k, so the profit is the manipulation. MSFT is seed-unstable (+$3,628 to +$15,886); INTC is stable.
Structurally similar to Coscia in the dimensions the agents choose (fills 0.4–1.4% vs his 0.08–0.5%, side
alternation); different in speed (44–52% of spoofs rest over 1 s vs his 0.57%).

**Sensitivity.** In the synthetic agent-based market, where price emerges from order matching rather than a
fitted impact model, a spoof moves the price 4.5–9× less and **every manipulator loses money**. Profitable
manipulation is established only under the replay impact model. Which model is closer to reality is unresolved:
the synthetic market's only reactive channel is imbalance-following, so it is plausibly too inert — a hypothesis,
not a defence.

**Simulator accounting — no defects found.** `tests/test_env_invariants.py`, 10 tests on AAPL and INTC: P&L
identity `cash + inventory × unimpacted mid` holds every step; every cash movement is explained by a trade,
run-over or liquidation; episodes end flat; duplicate spoofs ignored and cancels idempotent; refused orders
change nothing; reward comparable across a $585 and a $27 stock; observations inside the declared Box.

**Not simulated:** queue position (run-over uses "price moves through the level"), exchange fees, latency, and
historical prices never react to the agent. One large order per side is permitted, so layering cannot be learned
— only scripted, in the synthetic market.

---

## G. Test report

**191 passing, 24 test files, ~69 s.** No tests were disabled, weakened or skipped to obtain a pass.

| Area | File | Tests |
|---|---|---|
| Environment behaviour | `test_env.py` | 25 |
| Price impact & calibration | `test_price_impact.py` | 15 |
| Data loading & alignment | `test_data.py` | 11 |
| Synthetic order book | `test_synthetic_market.py` | 11 |
| Feature normalization | `test_normalization.py` | 10 |
| Metric consistency | `test_metric_consistency.py` | 10 |
| Simulator invariants | `test_env_invariants.py` | 10 |
| Synthetic env adapter | `test_synthetic_env.py` | 8 |
| Watchdog env & flagging | `test_watchdog_env.py` | 7 |
| Episode recording | `test_rollout.py` | 7 |
| Synthetic simulator | `test_synthetic_simulator.py` | 6 |
| Label observability | `test_label_observability.py` | 6 |
| Causal labels | `test_causal_labels.py` | 7 |
| Temporal split | `test_time_split.py` | 4 |
| Rule detector | `test_baseline_detector.py` | 4 |
| Roster granularity | `test_roster.py` | 3 |
| Multi-seed CIs | `test_multiseed.py` | 3 |
| Layering, ablation, evaluator | 3 files | 6 |

**Explicitly NOT verified, and why:**
- **End-to-end training convergence** — far too slow for a test suite. CP-7 models were trained once at 1M steps.
- **The Streamlit UI** — verified manually in a browser (all three views, the new tab, selector and limitation
  cards), not by automated browser tests.
- **`data/watchdog_ext`, DRIP and REVERSAL recorded datasets** — gitignored and built on Arnav's machine. They
  were regenerated here to re-enable full re-scoring; see §I.
- **Synthetic-market realism** — a modelling question, not a correctness one.

---

## H. Dashboard summary

Three views, every figure read from committed files in `results/` and `configs/`; nothing is trained, simulated
or fetched live.

- **Overview** — the one-sentence purpose, four KPI cards, and a warning bar carrying §C so no detector number is
  read without it.
- **Demonstration** — a recorded episode replayed step by step with the order-book ladder, flags and P&L. No
  live market connectivity is implied or faked.
- **Evidence & limitations** — five tabs led by *"Is the task hard?"*, which shows the zero-parameter signature
  and the supervised comparison before any headline result; then headline evidence, held-out tests, robustness
  and technical detail, closing with six limitation cards.

Files: `app.py` (Streamlit console), `scripts/build_dashboard.py` → `dashboard.html` (static one-pager).
Launch requires the file-watcher flag, because the repo package `watchdog/` shadows the PyPI `watchdog` library:

```bash
streamlit run app.py --server.fileWatcherType none
```

---

## I. Reproducibility guide

Verified from a fresh clone: **12/12 scripts reproduce exactly — 2,178 numeric values, 0 differences**, plus all
32 recorded dataset files bit-for-bit and `configs/calibration.json` identical.

```bash
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
python data/download.py                                      # ~600 MB LOBSTER data
python scripts/calibrate.py
python -m pytest tests -q                                    # 191 tests, ~69 s
python watchdog/dataset.py --episodes 150                    # record surveillance data
python watchdog/dataset.py --extended                        # CP-7 round-2 training sources
python watchdog/evaluate_watchdog.py --tag cov               # the CP-7 detector
python scripts/label_observability.py                        # the headline finding (C)
python watchdog/supervised_baselines.py train                # is RL needed? (D3)
python scripts/compare_protocols.py                          # what the split defect cost (B1)
python watchdog/posthoc_rescore.py                           # all generations, all datasets
streamlit run app.py --server.fileWatcherType none
```

**Dependencies.** `requirements.txt` is the declared set; `requirements-lock.txt` is a `pip freeze` of the
macOS environment that reproduced every one of Arnav's results and is left exactly as recorded. One defect was
found and fixed during this audit: `scripts/supervised_baseline.py` imported scikit-learn without declaring it,
so a fresh clone following the README hit a bare `ImportError`. scikit-learn is now declared, and the script
exits with an actionable message if it is absent. It is the only module in the repo that needs it, which is why
the lock file (predating that script) does not list it.

**Artifacts not in git** (`checkpoints/`, `data/` are gitignored): model checkpoints and recorded datasets must
be rebuilt or copied. All CP-7 checkpoints are present on this machine and were used for §D1.

---

## I2. Final quality gate — every check actually run

| Check | Command | Result |
|---|---|---|
| Full test suite | `python -m pytest tests -q` | **191 passed**, 69 s |
| Module imports | 13 core modules imported individually | all OK |
| Entrypoint smoke | `--help` on every script with a `main()` | 5 failures found and fixed (see below) |
| Hardcoded local paths | grep for `C:/Users`, `/home/`, `/Users/` in sources | none |
| Secrets / credentials | grep for api keys, tokens, passwords, private keys | none |
| Oversized tracked files | every tracked file > 5 MB | none; `.git` is 29 MB |
| Dashboard startup | Streamlit launched headless, `/_stcore/health` polled | serves HTTP 200 |
| Working tree | `git status --porcelain` | clean, in sync with origin |
| Cross-platform reproduction | DRIP + REVERSAL rebuilt and re-run | 0 real differences (§D0) |

**The entrypoint smoke test is the one that earned its place.** `--help` on `watchdog/drip_test.py`,
`reversal_test.py`, `lateburst_test.py` and `scripts/real_flow_rule.py` *executed the test* rather than
printing usage, because those scripts took no arguments and ignored argv — rewriting committed single-run
results. One of them, `real_flow_rule.py`, produced a **less complete** file than the committed one (the SPY
block needs data absent from this machine); it was restored from git. `scripts/pnl_decompose.py --help`
crashed with an `IndexError`. All five are fixed and guarded by `tests/test_cli_guard.py`.

**Not run:** linting and static type checking (no linter config in the repo, and adding one now would
produce churn rather than evidence); dependency CVE scanning (no scanner available in this environment).

## J. Final research verdict

**What the project now establishes.**
1. RL agents, rewarded only for profit, independently discover spoofing on narrow-spread stocks, and the profit
   comes from the manipulation — confirmed by an impact-free counterfactual across 3 seeds.
2. A labelling choice can silently redefine the task. The hindsight label made the detector a *predictor of
   actors*; the causal label made it a *detector of acts*, and that single change moved held-out recall from
   0.22 to 0.98 across four independently pre-registered attack families.
3. Pre-registration changes conclusions. Three attacks were registered before their data existed; the first went
   against the hypothesis and was reported that way.
4. Detector performance on this task is bounded by a zero-parameter rule, and a supervised LSTM reaches that
   bound. **Reinforcement learning is not justified for the detector.**

**What it does not establish.**
- Any real-world spoofing detection capability. No labelled real manipulation data exists; LOBSTER carries no
  participant identities, so the Watchdog cannot even be run on real order flow — only the rule can, and only
  as a flag rate.
- That spoofing detection is hard. In this formulation it is not (§C).
- That spoofing is profitable in a real market. It is profitable under the replay impact model and loses money
  under an order-matching one.
- Anything about layering learned rather than scripted.
- **Why** attribution-free detection works on INTC and MSFT but not AAPL, GOOG or AMZN. Two mechanical
  explanations were tested and refuted; five instruments cannot support a third
  (`results/detectability_by_instrument.md`).

**What real-world validation would require,** in order: participant attribution on real order flow (the binding
constraint); labelled or adjudicated manipulation cases; multiple trading days and market regimes; and a
detector evaluated without the label's definition in its input.

**Verdict.** As a research artefact this is sound: correct, reproducible, independently verified across two
machines, pre-registered where it matters, and honest about a finding that undercuts its own headline. Its
weakest point is not an error but a framing — the title promises detection, and what the project delivers on
that side is a well-executed negative result alongside a genuine positive one on the attacker side.
