# Changelog — audit and improvement cycle

Chronological, with the commit that made each change. Earlier project history is in `git log` and
`HANDOFF.md` §6.

---

## Branch reconciliation

**`5a32de9` — merge Arnav's research branch into main.** Two lines of work had diverged from `d9e242b`: 19
commits on `origin/arnav` (the CP-7 detector research) and 17 on `main` (protocol audit, dashboard,
documentation). Three files conflicted — `watchdog/dataset.py`, `README.md`, `.gitignore` — and all three were
resolved by keeping both sides. `TIME_SPLIT`/`--time-split` and `EXT_DATA_DIR`/`--extended` now coexist;
`causal_labels` and `split_window` are both present.

Both tips are preserved as tags: `archive/arnav-branch`, `archive/pre-merge-main`.

**Both work streams independently found the same core problem.** Arnav's LABEL-RULE ("the definition of the
label written as a rule … the ceiling, perfect by construction, not a competitor") and this line's
zero-parameter signature are the same finding reached separately.

---

## Confirmed defects fixed

| Commit | Defect | Measured impact | Regression test |
|---|---|---|---|
| `4368927` | Episode starts drawn from the whole day, so splits shared market data — 33 of 50 test episodes overlapped training (mean 980 of 2,000 events) | **0.031** order F1 in-distribution; held-out unaffected | `tests/test_time_split.py` (4) |
| `35cc436` (Arnav) | Hindsight labels marked every resting step, so training rewarded predicting *who would spoof* over detecting *the act* — 99% of SPOOFER-04's orders flagged before any opposite trade | held-out recall 0.22–0.30 → **0.98–0.99** | `tests/test_causal_labels.py` (7) |
| `b70712e` | `scripts/supervised_baseline.py` imported scikit-learn without declaring it anywhere; a fresh clone hit a bare `ImportError` | reproducibility | declared + actionable error |

---

## Verification performed

| Commit | What was checked | Result |
|---|---|---|
| `ca66047` | Arnav's `cov` result re-run on Windows from his macOS-trained checkpoint | **0.984 / 0.001 / 0.991 / 0.983** — matches his committed values to 3 d.p. |
| `155de8f` | DRIP and REVERSAL datasets rebuilt from scratch, pre-registered tests re-run | 1,164 and 1,615 numeric values, **0 real differences**; five per-stock P&L figures differ at relative ~1e-15; all verdicts bit-identical |
| `59991c3` | — | *(A correction: an earlier claim of "1,164 values, 0 differences" had been measured against a file that `git add -A` had already updated, so it compared the regenerated file with itself. Re-measured against a pre-regeneration backup; conclusion unchanged and stronger.)* |

---

## New experiments

**`29a22e0` — detection without participant attribution.** The project's largest open item.
`watchdog/anonymised_experiment.py` withholds the entire participant block and detects from the displayed
order book alone.

- Leakage audit: withheld pair step F1 **0.948**; best retained single-feature rule **0.490**; largest
  retained correlation **0.381**; retained features above 0.5: **0**. The removal is real.
- Step level, held-out: **F1 0.392** (logistic regression) vs **0.218** for the best constant — a real result.
- Episode level, held-out: best model 0.753 vs constant **0.769** — reported as **not a result**, because the
  set is 62.5% positive and flagging everything wins.
- Bounding limitation recorded: only one participant acts on top of replay here, so book anomalies are
  attributable by construction. These are an **upper bound** on attribution-free detection.

**`a03946f` — label observability.** Zero-parameter signature: order F1 **0.997** in-distribution, **0.995**
held-out, FPR 0.000. Re-measured on clean temporal splits: 0.995, so not an artefact of the split overlap.

**`4c7201c` — protocol comparison.** Three conditions isolate the split defect's cost from the model's:
original/original 0.772, clean/original 0.741, clean/retrained 0.769.

---

## Documentation reconciled

**`ca66047`** — the merge left the README telling two stories at once. All four documents and both dashboards
now give one account: the original Watchdog failed; the cause was the label; the fix worked across four
held-out attack families; and the ceiling it reached is occupied by a rule with no parameters.

- `docs/FINAL_AUDIT.md` (moved from `AUDIT.md`) — restructured as deliverables A–J.
- `docs/EXPERIMENT_PROTOCOL.md`, `docs/MODEL_COMPARISON.md`, `docs/DEMO_GUIDE.md`, this file — new.
- `README.md` — claim 2 rewritten as original → diagnosis → CP-7 → caveat; headline tables span both
  generations plus the label ceiling.
- `REPORT.md` — new §6.1b on the causal label; abstract and discussion reframed.
- `HANDOFF.md` — new §2c on how the detector was fixed and what the fix does and does not prove.
- `app.py` — KPIs read CP-7; new "Detector generations" tab; "Is the task hard?" carries the full ceiling
  comparison. Verified rendering in a browser.

---

## Tests

| | Before | After |
|---|---|---|
| Passing | 115 | **214** |
| Files | 15 | 25 |

Added: time-split integrity (4), metric consistency (10), simulator accounting invariants (10), label
observability (6), causal labels (7), attribution-leakage guards (7), CLI overwrite protection (12). No test was disabled, weakened or
skipped to obtain a pass.

---

## Engineering defects fixed late in the cycle

Found by running every entrypoint's `--help` as a smoke test, which is how they surfaced:

- **`--help` executed pre-registered tests.** `watchdog/drip_test.py`, `reversal_test.py`,
  `lateburst_test.py` and `scripts/real_flow_rule.py` took no arguments, so they ignored argv and ran.
  `--help` therefore re-ran a single-run test and rewrote its committed result. `evaluation/cli.py` now
  prints usage for `--help`, rejects unknown arguments with exit 2, and refuses to overwrite existing result
  files without `--rerun`. Guarded by `tests/test_cli_guard.py` (12 tests), including one that fails if a
  single-run script is ever added without the protection.
- **`scripts/real_flow_rule.py` was re-run by that accident and dropped the SPY block** (that data is not on
  this machine), producing a *less* complete file than the committed one. Restored from git; the incident is
  why the overwrite protection exists.
- **`scripts/pnl_decompose.py --help` crashed** with an `IndexError` from indexing empty positional args. It
  now prints usage, validates its arguments and exits 2 on misuse.
- The three pre-registered tests that re-ran produced **byte-identical results**, which is an unplanned
  reproduction check in their favour.

## Reporting gaps closed in the attribution-free experiment

The experiment had been run single-seed and reported only pooled precision/recall/F1/FPR. Eight gaps closed:

- **Multi-seed** (`--seeds 0 1 2`) with 95% Student-t intervals using small-sample t values, not 1.96.
- **PR-AUC** was computed but never rendered. It is the fairer headline here: against a held-out base rate of
  0.096, logistic regression reaches 0.314 and gradient boosting 0.321.
- **Confusion matrices** per model on the held-out split.
- **Per-source breakdown**, which gives instrument and attack family at once.
- **Supervised LSTM** over whole episodes, the fair comparison for a recurrent RL detector.
- **Calibration** — Brier score plus a reliability curve, against the base-rate reference.
- **Feature dictionary** (`docs/FEATURE_DICTIONARY.md`), with every index verified against the code.
- **12 tests** (`tests/test_anonymised_metrics.py`) pinning each statistic to a hand-checkable case.

**The breakdown changed the conclusion more than the headline F1 did:**

1. The LSTM is the best in-distribution model (step F1 0.864 ± 0.020, PR-AUC 0.933 ± 0.026) and the worst
   out of it (F1 0.052 ± 0.011). Reporting only in-distribution numbers would have crowned it.
2. Detection works **only on the two 1-tick stocks**: logistic regression reaches F1 0.585 (INTC) and 0.600
   (MSFT), but 0.073–0.131 on AAPL, GOOG and AMZN, where it *loses to a constant*. The pooled 0.392 averages
   two opposite results.
3. **No model is calibrated out of distribution** — all three score a worse Brier than the base-rate constant
   (0.110–0.139 against 0.099), despite beating it comfortably in-distribution.

## Follow-through after the gap closure

**P&L suppression re-measured with CP-7** (`results/posthoc/cov/pnl_suppression.md`). The headline card had
been showing the *original* detector's 83% beside CP-7 detector metrics. Re-run: CP-7 drives SPOOFER-04 from
+$12,625 to **−$133 per episode (101% suppressed)** against the rule's 55%, and intervenes **0.0 times per
episode** on legitimate FLICKER orders against the rule's 4.7. Suppression above 100% means the attacker ends
up losing money, so the P&L column is the honest figure, not the percentage. The attackers are frozen and do
not adapt to being watched.

**Why the per-instrument split exists: still unknown, and now documented as such**
(`results/detectability_by_instrument.md`, `scripts/detectability_by_instrument.py`). Two mechanical
explanations were tested against the data and **both refuted**:

- *The spoof is proportionally larger where it is detected.* No — the attacker places 10.00× touch depth on
  all five stocks by construction.
- *The spoof moves the book more, or further relative to its noise.* No — the shift in normalised deep
  imbalance is **smaller** on INTC and MSFT than on AAPL and AMZN, and signal-to-noise correlates −0.06 with
  detectability, ranking the stocks in a different order entirely.

What remains is a descriptive association with spread width (−0.84) across five instruments, which cannot
support a mechanism; tick-size regime and spread width coincide here and cannot be separated. Seven tests
(`tests/test_detectability_by_instrument.py`) fail if either refuted mechanism ever becomes explanatory, so
the write-up cannot silently go stale.

## Known not-done

- The attribution-free experiment uses **supervised models only**; no RL detector was retrained on it.
- Ablations beyond features (recurrent vs feed-forward, reward variants) not run.
- *(Corrected: the layering test **was** re-measured for CP-7, 3 seeds, in `results/posthoc_summary.md`.
  An earlier entry here claimed it had not been. It had — by Arnav, before the merge.)*
- Non-profitable Spoofers remain single-seed.
- One trading day, five stocks.
