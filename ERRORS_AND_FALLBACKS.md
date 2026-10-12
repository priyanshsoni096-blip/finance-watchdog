# Errors and fallbacks

For every feature or addition: how it can fail, how the failure shows up, and what to do instead. Add an entry **before** building a new feature, not after it fails. When a fallback is actually used, log it in [UPDATES.md](UPDATES.md) and reference the feature id.

Entries F-01 to F-09 cover features that already exist; their failure modes were observed during the project (`README.md`, `HANDOFF.md` §6). Entries for planned work are added once the execution plan arrives.

**Entry template**

```
## F-NN — <feature>
- Fails if: <concrete failure mode>
- How it shows: <the measurement or symptom that reveals it>
- Fallback 1: <first thing to try>
- Fallback 2: <if that fails>
- Last resort: <drop / reframe the claim / report as a negative result>
- Status: untested | fallback used (U-NNN) | not needed
```

---

## Existing features

### F-01 — Data download (`data/download.py`)
- Fails if: the Hugging Face repo `totalorganfailure/lobster-data` is unreachable or files change.
- How it shows: HTTP error, or `tests/test_data.py` fails on the first-row values it checks against.
- Fallback 1: download the sample files directly from lobsterdata.com (same 2012-06-21 sample, same file names).
- Last resort: no other day exists in the sample release, so the project is blocked without this data.
- Status: not needed (2026-10-10). All 10 files downloaded from Hugging Face; first-row checks in `tests/test_data.py` pass.

### F-02 — Environment setup on macOS
- Fails if: a pinned dependency has no wheel for Python 3.13 on macOS, or results differ on macOS because the project was built on Windows.
- How it shows: `pip install` errors, or `basic_eval.py` output that does not match `results/basic_eval.json` (479 numbers).
- Fallback 1: a Python 3.12 virtual environment.
- Fallback 2: if numbers differ, compare per stage (calibration → dataset bit-for-bit → evaluation) to find where they diverge. Report platform differences rather than overwrite committed results.
- Note: the committed `.md` reports contain Windows paths (`checkpoints\WATCHDOG\...`), so text diffs are expected even when numbers match. Compare the `.json` files, ignoring `runtime_s`.
- Status: not needed (2026-10-10). Reproduced on macOS/M4 with newer packages (U-015). `requirements.txt` is unpinned, so `requirements-lock.txt` now records the exact working versions. If a future install breaks, install from the lock file.

### F-03 — Spoofer training (`training/train_spoofer.py`)
- Fails if: the policy converges to never trading, or profit comes from an environment exploit.
- How it shows: trade fraction ≤ 0.2% of actions in `progress.csv`; or `scripts/pnl_decompose.py` shows profit that survives with `impact_lambda=0` or that does not come from spoof gain.
- Fallback 1: 10 events per decision plus entropy 0.01 (the fix that worked: INTC +$8,236 at 300k steps).
- Fallback 2: run trained Spoofers with sampled actions, not the most likely one (with the most likely action, SPOOFER-02 lost its spoofing).
- Last resort: use scripted attackers as the manipulation source and state that RL did not learn it on that stock (already the case for AAPL, GOOG, AMZN).

### F-04 — Watchdog training (`watchdog/train_watchdog.py`)
- Fails if: it collapses to never flagging.
- How it shows: flag rate 0.000 and recall 0 in `progress.csv` after about 30–60k steps.
- Fallback 1: `--norm-obs --ent-coef 0.01` (required; the defaults reproduce the collapse).
- Fallback 2: false-positive cost −1 instead of −2 (it also learned, with FPR ≤ 0.010).
- Last resort: report the supervised diagnostic (`scripts/feature_signal_check.py`) as an upper bound and the RL result as negative.

### F-05 — Watchdog generalisation to unseen attackers
- Fails if: recall on a new attacker is low (current state: 0.24 on LATEBURST, 0.30 on layering).
- How it shows: the pre-registered test result.
- Fallback 1: new order-history features or a sparse-style training population, chosen on test-split evidence only, then a *new* pre-registered attacker.
- Fallback 2: a hybrid where the rule supplies recall and the Watchdog's low FPR filters its flags (to be pre-registered before it is tried).
- Last resort: report Claim 2 as not shown. That is the current honest state and is publishable as a negative result.

### F-06 — Spoof profitability across market mechanisms
- Fails if: profit appears only under the replay impact model (current state).
- How it shows: in `results/synthetic_eval.md` every manipulator loses money.
- Fallback 1: add a second reactive channel to the synthetic market (market makers that reprice quotes) and re-measure emergent impact.
- Last resort: write Claim 1 as conditional on the impact model and present the 4.5–9× impact gap as a limitation.

### F-07 — Pre-registered tests
- Fails if: the test is run more than once, changed after results are seen, or uses an attacker that was already inspected.
- How it shows: this is a process failure, not a number.
- Fallback: none that rescues it. Discard the test, label any number from it post-hoc, and pre-register a fresh attacker.

### F-08 — Price-impact calibration
- Fails if: the fitted λ implies implausible moves.
- How it shows: `tests/test_price_impact.py` (a 10×-depth spoof must move the price 0.1–5 spreads; OFI R² > 0.2).
- Fallback 1: the `sqrt` impact form (`scripts/calibrate.py --form sqrt`).
- Recorded negative result: the percentile rule (44–58 spreads on MSFT/INTC) is not a fallback.

### F-09 — Synthetic market calibration
- Fails if: the market is frozen or liquidity grows without bound.
- How it shows: mid-change share near 0, or touch depth far above about 13,450.
- Fallback: a per-order cancel hazard and heavier-tailed market-order sizes (both already applied; earlier failures are recorded in `synthetic/calibrate.py`).

---

## Known issues in the inherited code (found 2026-10-10, not yet fixed)

| # | Issue | Where | Risk |
|---|---|---|---|
| K-1 | `.gitignore` excludes `checkpoints/`, but the README and HANDOFF also say the models are committed | `.gitignore`, `README.md` | Confusion on a fresh clone |
| K-2 | `REMOVED` has no `"intervention"` key, so serialising a surveillance-cancelled order would raise `KeyError` | `watchdog/dataset.py` | Latent; nothing triggers it today |
| K-3 | Stale comment says calibration does not constrain the follower share; the README retracted this | `synthetic/spoof_impact.py` (`--p-follower` argument) | Misleading for readers |
| K-4 | Committed reports contain Windows paths | `results/*.md` | Spurious diffs when regenerated on macOS |
| K-5 | `basic_eval.py --episodes 20` is listed twice in "How to run" | `README.md` | Cosmetic |
| K-6 | Fresh-clone reproduction check never finished | `HANDOFF.md` §8 | Some results not independently re-verified |

---

## Planned features (from `execution-plan-10.md`, mapped in `CHECKPOINTS.md`)

### F-10 — Spoofer training plots and behaviour summary (CP-2)
- Fails if: `progress.csv` lacks a needed column, or logs for the earlier failed runs were never kept.
- How it shows: missing columns or missing folders under `checkpoints/`.
- Fallback 1: plot what exists (reward, PnL, action mix, trades against own spoof) and take behaviour statistics from `results/real_case_stats.json`.
- Fallback 2: for the failed runs, cite the numbers already in the README findings rather than re-train.
- Last resort: a table instead of plots, with the limitation stated.

### F-11 — Consolidated results table (CP-3)
- Fails if: a bucket's JSON lacks the confusion counts, or buckets are not defined identically across reports.
- How it shows: missing keys, or totals that do not add up between reports.
- Fallback 1: re-run only the evaluation script for that bucket (the data are seeded, so it regenerates identically).
- Fallback 2: show the bucket in a separate table with a note on how it differs.
- Note: the rule tuned on held-out data is a sensitivity row only. It never replaces the train-tuned rule, which is the pre-registered comparison.

### F-12 — SPY rule-only real-flow test (CP-4)
- Fails if: the SPY download (~940 MB) fails, or the 30-level files do not truncate cleanly to 10 levels.
- How it shows: download error, or `tests/test_data.py`-style sanity checks failing on SPY.
- Fallback 1: run the rule directly on the 30-level message stream (the rule uses messages, not book levels).
- Last resort: drop SPY and report the five-stock real-flow flag rates already in `results/basic_eval.md`, as the README already does.
- Fixed limitation (no fallback exists): the Watchdog cannot run on real flow without participant identities.

### F-13 — Multi-seed PnL suppression (CP-5)
- Fails if: runtime is too long, or seed variance is so large the CI is uninformative.
- How it shows: run time per seed (main took 50 episodes × 3 conditions per source), or a CI that crosses zero suppression.
- Fallback 1: restrict to the profitable RL Spoofers (SPOOFER-02, SPOOFER-04), the only sources where suppression is meaningful.
- Fallback 2: more episodes per seed before more seeds.
- Last resort: report the wide CI as the result (3 seeds; Student-t).

### F-14 — Concrete flag explanations (CP-6)
- Fails if: per-step attribution is noisy, or explanations contradict the group ablation.
- How it shows: attributions that change sign between adjacent steps, or a top feature other than trade direction with no plausible reason.
- Fallback 1: occlusion per feature group over the order's resting window (consistent with `watchdog/feature_ablation.py`), instead of per-step gradients.
- Fallback 2: present the order timeline (placement, trades, flags, cancel) with the group-ablation result as the explanation.
- Note: the Watchdog has no attention layer, so the plan's attention option does not apply.

### F-15 — Improving detection of sparse spoofing (CP-7)
- Fails if: new features or training sources raise test-split recall on sparse spoofing but not on the new pre-registered attacker, or raise false positives.
- How it shows: the single pre-registered run; FPR on legitimate orders above the current 0.031 ± 0.067.
- Fallback 1 (revised after U-017): a causal training label, positive only from the first opposite trade while the order rests. This rewards evidence, not guessing the trader's identity. Chosen on test-split evidence only.
- Rejected: a feature counting opposite trades against the resting order. It encodes the label's own definition (label leakage) and would undercut the "never told the rule" claim.
- Fallback 2: add sparse-style scripted sources with randomised waits and trade counts to training (never the attackers used for held-out tests, nor the new pre-registered one).
- Fallback 3: rule-then-Watchdog hybrid (pre-registered before it is tried).
- Last resort: Claim 2 stays "not shown". That is a valid, reportable negative result.
- Status (2026-10-12): fallback 1 used (U-019 to U-022). It fixed in-distribution sparse detection (GOOG recall 0.056 → 0.997) but not timing generalisation (DRIP-ATK recall 0.26). Fallback 2 is next. When writing its scripted sources, vary the wait (including 10–40 steps) and add acts on old orders; avoid DRIP-ATK, LATEBURST and SCRIPTED parameters, and pre-register a fresh attacker.

### F-16 — Write-up (CP-8)
- Fails if: a number cannot be traced to a committed file, or the headline the plan expected (Watchdog ahead on unseen attacks) does not hold.
- How it shows: a claim in the draft with no source; the current results.
- Fallback 1: re-run the script behind any untraceable number, or remove the number.
- Fallback 2: lead with what does hold: PnL suppression (83% vs 55%) and false positives about 26× lower. Report the generalisation result as negative, in the same table. The plan itself says failed runs and negative results are content, not holes.
