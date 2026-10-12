# Updates

A running log of everything done on the project, kept so it can be turned into a research paper. Newest entries go at the bottom of the log.

**Rules (the same ones `HANDOFF.md` §9 uses):**
- Every number names its source: a committed `results/` file, or the script and command that printed it. Nothing is written from memory or estimated.
- Negative results, failed runs and rejected changes are logged as fully as successes.
- Any result that came from looking at held-out data is labelled post-hoc.
- Pre-registered tests are logged twice: once when the attacker, procedure and hypothesis are committed (before any data exists), and once with the single result.

**Entry template**

```
### U-NNN — <title> (<YYYY-MM-DD>)
- Type: setup | method | experiment | result | fix | decision | negative result | pre-registration
- Paper section: Methods | Results | Discussion | Limitations | Appendix
- What and why: <one or two sentences>
- Result: <numbers, with source>
- Checkpoint: <CP id> · Fallback used: <feature id or none>
```

---

## Part A — Backfill: work done before 2026-10-10

Reconstructed from `README.md` and `HANDOFF.md`. The original dates were not recorded, so these entries are ordered by project phase. They are summaries for the paper: the detail stays in the cited files.

### U-001 — Research question and framing
- Type: method · Paper section: Introduction
- Can an RL Watchdog detect previously unseen spoofing with a low false-positive rate, without being told the rule? The required disclaimer sentence and the CEA §4c(a)(5)(C) reference are in `README.md`. No labelled real-world spoofing data exists, so the project never claims real-world detection.

### U-002 — Data and cross-stock normalisation
- Type: method · Paper section: Methods
- LOBSTER sample day 2012-06-21, five stocks (AAPL, MSFT, GOOG, INTC, AMZN), 10 levels. Prices are in units of the trailing spread and sizes are log ratios to a trailing size, because spreads range from 1 tick (INTC) to 15 ticks (AAPL).
- Source: `env/normalization.py`; the README "Measured calibration" table.

### U-003 — Price impact fitted from order-flow imbalance
- Type: method + negative result · Paper section: Methods, Appendix
- λ is fitted per stock from order-flow imbalance (Cont, Kukanov & Stoikov 2014). OFI R² ranges from 0.33 (AMZN) to 0.73 (MSFT). The handoff's percentile rule was rejected because it implied a 10×-depth spoof moves MSFT/INTC by 44–58 spreads.
- Source: `configs/calibration.json`; README design change 4.

### U-004 — Four environment exploits found and closed
- Type: fix · Paper section: Methods (environment design), Appendix
- Free accumulation (about +$253k per episode), the alternating-side swing (about $240k per episode), escape through early termination (31/31 terminations on a run-over step), and a thin opening book. Each was found by decomposing where the PnL came from.
- Source: `HANDOFF.md` §6; README design changes 7–11.

### U-005 — Spoofers converged to never trading; fixed with 10 events per decision and entropy 0.01
- Type: negative result + fix · Paper section: Methods, Discussion
- A scripted probe showed spoofing was profitable (+$3,980 INTC, +$4,043 MSFT), so the failure was credit assignment, not a missing opportunity.
- Source: README "Findings so far"; design change 12.

### U-006 — Final Spoofer population
- Type: result · Paper section: Results (Claim 1)
- SPOOFER-04 INTC +$13,208 ± 1,654 and SPOOFER-02 MSFT +$2,437 ± 1,068 per episode. With impact switched off, the same trading loses (−$8,414 and −$5,407). AAPL, GOOG and AMZN are indistinguishable from zero.
- Source: `results/basic_eval.md`; `scripts/pnl_decompose.py SPOOFER-04 main 10 --stochastic`.

### U-007 — Watchdog collapsed to never flagging; fixed by normalising observations
- Type: negative result + fix · Paper section: Methods, Appendix
- It collapsed at entropy 0, 0.01, 0.03 and 0.1. The cause was input scale, not the false-positive cost.
- Source: README "Findings so far"; design change 14.

### U-008 — Watchdog vs rule, in distribution (3 seeds)
- Type: result · Paper section: Results (Claims 2, 3)
- Order F1 0.753 ± 0.058 vs 0.352. Legitimate-order FPR 0.031 ± 0.067 vs 0.823.
- Source: `results/multiseed.md`.

### U-009 — Pre-registered LATEBURST-ATK test
- Type: pre-registration + result · Paper section: Results (Claim 2)
- Watchdog recall 0.24 vs rule 0.89. The stock-shortcut hypothesis was not supported (ratio 1.50 against a threshold of 2).
- Source: `results/lateburst_test.md`.

### U-010 — PnL suppression, feature ablation, real-case comparison
- Type: result · Paper section: Results, Discussion
- The Watchdog cuts SPOOFER-04's profit by 83% (rule: 55%). Ablating trade direction drops F1 from 0.772 to 0.113. The learned behaviour is structurally similar to Coscia's but differs in speed and has no layering.
- Source: `results/pnl_suppression.md`, `results/feature_ablation.md`, `results/real_case_comparison.md`.

### U-011 — Synthetic agent-based market and the pre-registered layering test
- Type: method + result · Paper section: Results (robustness), Limitations
- Emergent spoof impact is 0.29–0.55 spreads vs about 2.5 in the replay model, and every manipulator loses money there. Detection's FPR advantage carries over (0.11 vs 0.92). Layering test: Watchdog F1 0.44 vs rule 0.73.
- Source: `results/synthetic_spoof_impact*.md`, `results/synthetic_eval.md`, `results/layering_test.md`.

---

## Part B — Log from 2026-10-10

### U-012 — Full repository review on a new machine (2026-10-10)
- Type: setup · Paper section: none (process)
- Read every source file, test, doc, config and result report. Local state: macOS, Python 3.13.3, not a git repository, no data downloaded, dependencies not installed.
- Inconsistencies found (not yet fixed): see `ERRORS_AND_FALLBACKS.md`, "Known issues in the inherited code".
- Checkpoint: CP-0.

### U-013 — Tracking files created (2026-10-10)
- Type: setup · Paper section: none (process)
- Created `CHECKPOINTS.md`, `UPDATES.md` and `ERRORS_AND_FALLBACKS.md`. Milestones and new fallbacks wait for the execution plan.

### U-014 — Execution plan received and audited against the repo (2026-10-10)
- Type: decision · Paper section: Methods (deviations from plan), Limitations
- `execution-plan-10.md` (dated 2026-09-01) added to the repo. Phases 0–7 are built, Phases 8–9 are partial, and Phase 10 is not started. Gaps and milestones are in `CHECKPOINTS.md` (CP-1 to CP-8). Fallbacks for the planned work are F-10 to F-16.
- Deviations from the plan, each already justified in the README (the paper's Methods should list them):
  - Prices in spread units, not ticks/basis points (design change 2).
  - Impact fitted from order-flow imbalance; the plan's percentile rule was implemented and rejected (design change 4).
  - Watchdog observation is 46-dim (40 book + 6 participant), not 43; it uses an LSTM over the whole episode, not a fixed 20-step window.
  - Agents decide once per 10 market events (design change 12).
  - Synthetic market calibrated on MSFT/INTC, not the pooled four stocks the plan locked.
  - The layering attacker exists only in the synthetic market; the replay environment allows one large order per side.
  - SPY not used; the Watchdog cannot run on real flow (no participant identities).
- The plan's target result (Watchdog ahead of the rule on held-out and scripted attacks) was not reached. The plan says failed runs and negative results should be reported, not hidden.
- The plan cites `project-design-consolidated.md` (§3a, §6), which is still not in the repo.
- Checkpoint: CP-0.

### U-015 — Reproduction on a second platform (2026-10-10)
- Type: setup + result · Paper section: Methods (reproducibility), Appendix
- What and why: re-ran the pipeline on a different OS, CPU and library stack to show results don't depend on the original machine.
- Platform: macOS on Apple M4, Python 3.13.3. Packages: gymnasium 1.4.0, numpy 2.5.3, pandas 3.0.6, torch 2.14.1, stable-baselines3 2.9.0, sb3-contrib 2.9.0 (full list in `requirements-lock.txt`). The original results were produced on Windows 11, Python 3.13, CPU only.
- Results:
  - Tests: 115 passed (14 s).
  - `scripts/calibrate.py`: 92 values, largest relative difference 1.6e-14 (last-digit floating point; not byte-identical).
  - `evaluation/basic_eval.py --episodes 20`: 549 values vs `results/basic_eval.json`, largest relative difference 8.6e-14.
  - `watchdog/dataset.py --episodes 150` then `watchdog/evaluate_watchdog.py`: 282 values vs `results/watchdog_eval.json`, all exactly equal.
- Runtimes: basic_eval 9 s (54 s originally), dataset 38 s (213 s), Watchdog evaluation 20 s.
- Committed `results/` and `configs/` files were left unchanged; regenerated outputs went to a scratch folder.
- Decision recorded: CP-7 (improving detection of sparse/unseen spoofing) will be attempted before the write-up; no deadline.
- Checkpoint: CP-1 done. Fallback used: none.

### U-016 — Consolidated results table and oracle-rule sensitivity (2026-10-10)
- Type: result · Paper section: Results (headline table), Appendix
- What and why: the plan's single table of every bucket for both detectors, with confusion matrices. Built by the new `scripts/consolidate_results.py` from the committed JSON; nothing re-estimated. Output: `results/consolidated.md` / `.json`.
- LATEBURST data regenerated (`watchdog/lateburst_test.py`, seeded) to build the sensitivity row: 269 values matched `results/lateburst_test.json` (largest relative difference 8.3e-14). The committed report files were restored afterwards; the pre-registered verdict is unchanged.
- Sensitivity check from the plan's risk register: the rule re-tuned with hindsight on each bucket's own orders (an oracle upper bound for any (N, M) rule).
  - In distribution: the oracle rule still gets F1 0.35 (it keeps N = 2, M = 1000), against the Watchdog's 0.77. **The Watchdog's in-distribution advantage is not caused by a badly tuned rule.**
  - LATEBURST-ATK: oracle rule F1 0.46 (same parameters as the train-tuned rule), Watchdog 0.35. The Watchdog is behind on F1 as well as recall on the unseen attacker.
  - Scripted attacker: the oracle rule improves to F1 0.72–0.73 with M = 200 (train-tuned 0.65–0.66); the Watchdog is 0.26–0.38.
  - Even tuned on held-out AMZN legitimate orders alone, the rule still flags 73% of them; the Watchdog flags 1%.
- Checkpoint: CP-3 done.

### U-017 — CP-7 diagnosis: the Watchdog recognises trader style, not the manipulative act (2026-10-10)
- Type: experiment (diagnostic, test split only) · Paper section: Results (failure analysis), Discussion
- What and why: find out when, in a manipulative order's life, the main Watchdog flags. New script `scripts/flag_timing_diagnostic.py` (refuses held-out data). Each resting step of a manipulative order is classed as *before* its owner's first opposite trade, a *trade* step, or *after*.
- Result (`python scripts/flag_timing_diagnostic.py`, main Watchdog, test split):

  | Source | Flag rate before any opposite trade | On opposite-trade steps | After |
  |---|---|---|---|
  | SPOOFER-04 INTC | 0.990 | 1.000 | 0.981 |
  | SPOOFER-02 MSFT | 0.840 | 0.996 | 0.944 |
  | SPOOFER-03 GOOG | 0.008 | **0.137** | 0.054 |

  - 89.5% of caught orders were flagged *before* any opposite trade existed (median: 1 step before the first trade).
  - Order recall rises with the number of opposite trades: 0.43 (1 trade), 0.57 (2), 0.74 (3), 0.88 (4), 0.98 (5+).
  - Orders whose first opposite trade came after 5–9 steps: recall 4/24 = 0.17; after 1–4 steps: 0.81.
- Interpretation: on MSFT/INTC the Watchdog flags a large order as soon as it appears. That is only possible by recognising the dense Spoofers' trading style from history (consistent with the ablation: removing trade direction collapses recall). On GOOG it misses even the step where the incriminating trade happens (13.7%). It has not learned the conjunction "trade on the side opposite a resting large order".
- Likely cause: the training label is set with hindsight over the whole resting window, including steps before any evidence exists. Reward is maximised by predicting *who* will trade, which works for dense spoofers and fails for sparse ones.
- Methodological note for the paper: the manipulation label is defined by an observable act (an opposite trade while the order rests). An engineered feature counting such trades would leak the label definition and undercut "never told the rule", so CP-7 will not add one.
- Checkpoint: CP-7 (diagnosis).

### U-018 — Version control restored (2026-10-12)
- Type: setup · Paper section: none (process; needed for pre-registration timestamps)
- The working folder was attached to the original history (`origin/main` at d9e242b). All work since 2026-10-10 is committed on branch `arnav`. The only differences from `main` were this session's additions; every committed result and checkpoint file was identical.
- From here, CP-7's pre-registered attacker and hypothesis are committed before any of its data is generated.

### U-019 — CP-7 experiments 1–2: causal label, then γ = 0 (2026-10-12)
- Type: experiment (test split only) · Paper section: Methods (Watchdog training), Results, Appendix
- Code: `watchdog.dataset.causal_labels` and `train_watchdog.py --label causal` (positive only from the owner's first opposite trade; the defaults still reproduce the reported Watchdog). Also `--gamma`, `evaluate_watchdog.py --test-only`, and `watchdog/compare_candidates.py` (scores candidates on the test split only; never loads held-out or LATEBURST data). New unit test: 116 tests pass.
- Effect of the causal label on training positives: SPOOFER-03 GOOG keeps 63% of its hindsight-positive steps, MSFT 86%, INTC 93%.
- All runs: 400k steps, `--norm-obs --ent-coef 0.01`, seeds 7/11/23 as for `main`. Results in `results/cp7_label.md` (mean ± 95% Student-t CI, 3 seeds):

  | Config | Order F1 | Legit FPR | GOOG order recall | INTC order recall |
  |---|---|---|---|---|
  | main (hindsight label, γ 0.9) | 0.753 ± 0.058 | 0.035 ± 0.078 | 0.056 ± 0.240 | 0.887 ± 0.487 |
  | causal label, γ 0.9 | 0.783 ± 0.231 | 0.008 ± 0.017 | 0.003 ± 0.012 | 0.823 ± 0.727 |
  | causal label, γ 0 | **0.923 ± 0.166** | 0.010 ± 0.019 | 0.641 ± 1.364 | 0.993 ± 0.013 |
  | causal label, γ 0, miss cost 2 | 0.854 ± 0.288 | 0.070 ± 0.287 | 0.373 ± 1.346 | 0.994 ± 0.004 |

- Findings:
  - The causal label alone raises precision (0.82 → 0.95) and lowers legitimate FPR, but does not teach the act: GOOG recall falls to 0.003.
  - γ = 0 is the change that matters. A flag never affects later observations, so the optimal policy is myopic and γ > 0 only adds variance.
  - Outcomes are bimodal by seed. Causal + γ 0 seeds 11 and 23 reach F1 0.960/0.962 with GOOG recall 0.92/0.99 (legit FPR 0.009/0.018); seed 7 reaches F1 0.846 but GOOG recall 0.008. The act is learnable, but at 400k steps not reliably found.
  - Miss cost 2 is rejected: less stable (one seed at legit FPR 0.204) and no better on GOOG.
- Next: the same configuration at 1M steps, to test whether stuck seeds find the sparse act with more training.
- Checkpoint: CP-7 (development; no held-out data seen).
