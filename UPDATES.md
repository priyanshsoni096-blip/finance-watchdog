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

### U-020 — CP-7 experiment 3: 1M steps makes it reliable; the Watchdog now detects the act (2026-10-12)
- Type: experiment + result (test split only) · Paper section: Results, Discussion
- Causal label, γ 0, 1M steps, seeds 7/11/23 (`checkpoints/WATCHDOG/causal_g0_1m{,_s2,_s3}`; about 10 min each, 3 in parallel). `results/cp7_long.md`:
  - Order F1 **0.978 ± 0.015** (main: 0.753 ± 0.058); precision 0.967 ± 0.030; recall 0.990 ± 0.000.
  - Legitimate-order FPR **0.003 ± 0.009** (main: 0.035 ± 0.078).
  - SPOOFER-03 GOOG (sparse) order recall **0.997** in every seed (main: 0.056). MSFT 0.994, INTC 0.996.
- Flag timing (`python scripts/flag_timing_diagnostic.py --tag causal_g0_1m`), compared with U-017:
  - Caught orders flagged before any opposite trade: 0.6% (main 89.5%). Flagged on the trade step itself: 99.4% (main 6.1%).
  - Flag rate before the first opposite trade: GOOG 0.006, MSFT 0.002, INTC 0.007 (main: 0.008 / 0.840 / 0.990).
  - Recall on orders with a single opposite trade: 0.974 (main 0.429).
- Interpretation: with a causal label and a myopic objective, the Watchdog learned the conjunction "the owner trades on the side opposite its own resting large order". It does so from rewards alone, never told the rule.
- **Caveat for the paper (must be stated):** that conjunction *is* the definition of the proxy label. A hand-written rule using the label's own definition would score perfectly on these labels by construction. The meaningful comparisons are: (1) against the industry-style size + fast-cancel rule, which cannot see the owner's trades; (2) learning it without being given the rule; (3) generalisation to attackers never seen in training. The label stays a proxy for intent (Open issue 4).
- Decision (test-split evidence only): candidate for the pre-registered test is `causal_g0_1m` (seed 7), with `_s2`/`_s3` reported for variability, the same protocol as LATEBURST. No held-out or LATEBURST data has been loaded for any CP-7 model.
- Checkpoint: CP-7 (development complete; pre-registration next).

### U-021 — Pre-registration: DRIP-ATK test (2026-10-12)
- Type: pre-registration · Paper section: Methods (evaluation protocol), Results
- Committed and pushed before any DRIP-ATK/SAMESIDE data was generated for scoring: the attacker `DripSpoof`, the new legitimate control `SameSideTrader` (both `evaluation/agents.py`), `watchdog/drip_test.py`, and the hypotheses in README "Pre-registered test: DRIP-ATK".
- H1: Watchdog (`causal_g0_1m`) order F1 on DRIP-ATK + SAMESIDE + FLICKER ≥ rule F1. H2: Watchdog FPR on SAMESIDE ≤ 0.10.
- Disclosure: a 3-episode label-mechanics check on throwaway GOOG seeds 999–1001, with no detector scoring.
- Checkpoint: CP-7 (pre-registered; not yet run).

### U-022 — DRIP-ATK result (run once) and post-hoc diagnosis (2026-10-12)
- Type: result + negative result · Paper section: Results (Claim 2), Discussion, Limitations
- `python watchdog/drip_test.py`, run once after pre-registration commit 8688d51 (pushed). `results/drip_test.md`:
  - **H1 supported** (Watchdog F1 0.409 vs rule 0.336, pooled). **H2 supported** (SAMESIDE FPR 0.012).
  - But DRIP-ATK recall is 0.26 against the rule's 0.74; on DRIP-ATK alone the rule's F1 is 0.85 vs 0.41. H1 passed because of the rule's false positives (FPR 0.78). Lesson: state recall hypotheses directly.
  - Precision transfers: 19 false positives in 4,108 legitimate orders (rule: 3,273).
  - Context: original `main` Watchdog on the same orders, F1 0.132 and recall 0.084. Seeds `_s2`/`_s3` reach recall 0.438/0.462 with SAMESIDE FPR 0.117/0.100.
  - Per stock recall: AAPL 0.40, GOOG 0.42, AMZN 0.36, MSFT 0.08, INTC 0.06.
- POST-HOC (`scripts/drip_posthoc.py`, after the verdict; informs no verdict):
  - Act flag rate by order age at the act: 1–4 steps 0.55 (n = 11), 5–9 0.244, 10–19 0.193, 20+ 0.045.
  - Camouflage before the act: 0.152 with, 0.219 without (minor).
  - First act 0.238, later acts 0.132.
  - MSFT/INTC acts flagged at 0.029–0.078 at every age.
- Interpretation: CP-7 changed what the Watchdog detects (the act, not the trader) and kept false positives very low on unseen legitimate styles. It does not yet generalise across timing: the training population's acts nearly all come 1–4 steps after placement, and the 1-tick stocks' Spoofers only show dense, early trading. This points to training-population coverage (ERRORS_AND_FALLBACKS F-15, fallback 2) as the next lever. DRIP-ATK is now inspected and can no longer serve as a clean test.
- Checkpoint: CP-7 iteration 1 complete.

### U-023 — CP-7 round 2: training-population coverage (2026-10-21)
- Type: method + experiment (test splits only) · Paper section: Methods (training population), Results
- Correction to U-022: the RL Spoofers' data does contain acts on older orders, but nearly always after earlier acts on the same order. What was missing was a *first* act that comes late (1,511 of 1,536 manipulative test orders had their first act within 1–4 steps).
- New training-only sources (`evaluation/agents.py`), on AAPL/MSFT/GOOG/INTC only, built with `python watchdog/dataset.py --extended --episodes 150` into `data/watchdog_ext/` (original splits untouched):
  - VARIED-SPOOF: 1–6 opposite trades, each after a log-uniform 1–400-event wait, same-side noise trades with probability 0.2. 29% of its acts come at order age 20+ steps, against 8% in the RL Spoofers' data.
  - ACCUM: the legitimate counterpart. Same order sizes, ages and trade counts, but trades only on the same side as its order.
- Training: `--label causal --gamma 0 --extra-data data/watchdog_ext`, 1M steps, seeds 7/11/23 (`checkpoints/WATCHDOG/cov{,_s2,_s3}`), 20 sources.
- Results (`results/cp7_round2.md`, mean ± 95% CI, 3 seeds; round 1 = `causal_g0_1m`):
  - Original test split: order F1 0.984 ± 0.001 (round 1: 0.978 ± 0.015); legitimate FPR 0.002 ± 0.002 (0.003 ± 0.009).
  - VARIED-SPOOF test recall 0.994 ± 0.002 (round 1: 0.506 ± 0.100). First act at 10+ steps: 1.000 (round 1: 0.324). MSFT 0.995, INTC 0.992 (round 1: 0.52, 0.49).
  - ACCUM FPR 0.031 ± 0.026 (round 1: 0.057 ± 0.092).
- Caveat: VARIED-SPOOF/ACCUM test numbers are in-distribution for this training family. Generalisation needs a new pre-registered attacker that is structurally different from it.
- Decision (test splits only): `cov` (seed 7) is the verdict model for the next pre-registered test; `_s2`/`_s3` for variability.
- Checkpoint: CP-7 round 2 (development complete).

### U-024 — Pre-registration: REVERSAL-ATK test (2026-10-21)
- Type: pre-registration · Paper section: Methods (evaluation protocol), Results
- Committed and pushed before any REVERSAL-ATK/REVERT data was generated for scoring: `ReversalSpoof` (Coscia-style reversal, carrying the position into each new spoof), the legitimate boundary control `CancelThenTrade` (opposite trades only after cancelling), `watchdog/reversal_test.py`, and the hypotheses in README "Pre-registered test: REVERSAL-ATK".
- Verdict model `cov` (frozen in 44508ca). H1: recall ≥ 0.80 (the recall hypothesis is now stated directly, the lesson from U-022). H2: FPR on REVERT + FLICKER ≤ 0.10. H3: pooled F1 ≥ rule F1.
- Disclosure: a 3-episode label check on throwaway seeds, and one end-to-end smoke run of the script on throwaway seeds whose outputs were deleted unread.
- Checkpoint: CP-7 round 2 (pre-registered; not yet run).

### U-025 — REVERSAL-ATK result (run once): all three hypotheses supported (2026-10-21)
- Type: result · Paper section: Results (Claim 2 — headline), Discussion
- `python watchdog/reversal_test.py`, run once after pre-registration commit ca58bc1 (pushed). `results/reversal_test.md`:
  - **H1 supported:** recall on REVERSAL-ATK 0.981 (rule 0.964).
  - **H2 supported:** FPR on REVERT + FLICKER 0.006, 24 of 4,052 (rule 0.804, 3,256 of 4,052).
  - **H3 supported:** pooled F1 0.986 (rule 0.591).
  - Per stock recall 0.96–0.99, MSFT 0.99 and INTC 0.98 included.
  - Seeds (context): `cov_s2` recall 0.984, REVERT FPR 0.034; `cov_s3` 0.984, 0.049.
  - Same orders (context): round-1 `causal_g0_1m` recall 0.400; original `main` 0.121, FPR 0.07/0.045.
  - 21 of the 24 false positives are REVERT orders (the boundary case: opposite trades right after the cancel).
- What it shows, and what it does not:
  - The Watchdog's advantage on this unseen attacker is precision. The rule also catches 96%, but flags 80% of legitimate large orders.
  - The proxy label is defined by the act the Watchdog detects (U-020 caveat). The claim is that a detector learned from rewards alone, never told the rule, transfers to a structurally new attacker and a new legitimate boundary case.
  - REVERSAL-ATK is unprofitable on 4 of 5 stocks, so this tests detection of the structure, not of profitable manipulation.
  - Replay market only; synthetic-market and layering behaviour of the round-2 Watchdog are not yet measured.
- Path to this result, all to be reported:
  - Original Watchdog: LATEBURST/LAYER failed.
  - Round 1: causal label + γ 0. DRIP-ATK passed on a pooled metric with recall only 0.26.
  - Round 2: training coverage. REVERSAL-ATK passed, with recall stated directly.
  - DRIP-ATK and REVERSAL-ATK are now inspected and no longer clean.
- Checkpoint: CP-7 done.

### U-026 — Post-hoc re-scoring of the CP-7 Watchdog, and multi-seed PnL suppression (CP-5) (2026-10-21)
- Type: result (post-hoc) · Paper section: Results (robustness, economic result), Appendix
- Code: `--tag/--out` options for `synthetic/evaluate_synthetic.py`, `synthetic/layering_test.py` and `watchdog/pnl_suppression.py` (defaults unchanged). New `watchdog/posthoc_rescore.py` (all replay data on disk) and `watchdog/posthoc_summary.py`. Outputs: `results/posthoc/<tag>/`, `results/posthoc_rescore.md`, `results/posthoc_summary.md`.
- Consistency check: the original `main` seeds re-scored by the new script give the committed `results/multiseed.md` values (F1 0.753 ± 0.058, legitimate FPR 0.031 ± 0.067, LATEBURST recall 0.217 ± 0.156).
- Replay, mean of 3 seeds (`posthoc_rescore.md`), original → round 2 (rule):
  - Recall: SCRIPTED train stocks 0.257 → 0.991 (0.938); LATEBURST 0.217 → 0.984 (0.885); DRIP 0.067 → 0.989 (0.744); REVERSAL 0.139 → 0.983 (0.964).
  - Legitimate FPR 0.031 → 0.002 (0.820).
- Synthetic market, Watchdog trained on replay only (`posthoc_summary.md`):
  - All manipulators: F1 0.622 → 0.980 ± 0.011 (rule 0.736). Legitimate FPR 0.113 → 0.030 ± 0.108 (rule 0.917).
  - LAYER-ATK recall 0.300 → 0.990 ± 0.025 (rule 1.000); pooled F1 0.441 → 0.984 (rule 0.733).
- PnL suppression, 3 seeds each (CP-5):
  - SPOOFER-04 INTC: 87.5% ± 19.1 (original) → 101.4% ± 0.9 (round 2); rule 55.2%. With the round-2 Watchdog the Spoofer loses $178 ± 109 per episode, against +$12,625 unsupervised.
  - SPOOFER-02 MSFT: 243% ± 23 → 242% ± 3; rule 96%.
  - Wrongful interventions on FLICKER per episode: 0.55/0.49 (original) → 0.013/0.007 (MSFT/INTC); rule 4.7.
- Caveats: post-hoc for every attacker here (all inspected earlier). The VARIED-SPOOF training family was designed after DRIP-ATK was inspected, and SCRIPTED/LATEBURST share its structure. Only REVERSAL-ATK is a clean test of the round-2 model. The synthetic and layering rows for `main` are single runs.
- Checkpoints: CP-7 follow-ups done; CP-5 done.

### U-027 — CP-6: concrete explanations of Watchdog decisions (2026-10-21)
- Type: result · Paper section: Results (interpretability), Discussion
- `python watchdog/explain_flags.py` (new) → `results/flag_explanations.md`. Watchdog `cov`. Flag probabilities are read in one batched recurrent pass; a check confirmed `P > 0.5` matches `flag_episode`'s deterministic flags exactly. Occlusion uses the training mean (0 after normalisation), at the decided step or over all earlier steps. A counterfactual flips the trade's side. Cases are picked by fixed rules (first match), not by hand.
- A, SPOOFER-03 GOOG test: flag P = 1.000 on the opposite trade 7 steps after placement; P = 0.000 on all earlier steps. Trade direction (this step) −1.000; flipped side −1.000; all else ±0.000 (book, age, history).
- B, REVERSAL-ATK AAPL (pre-registered data): flag P = 1.000 on the opposite trade 25 steps after placement. Trade direction (this step) −1.000; flipped side −1.000.
- C, REVERT AAPL, false positive: P = 0.702 three steps after placing an ask, no trade that step. Drivers: trade-direction history −0.702, side-imbalance history −0.697, resting size −0.695. Cause: a buy 4 steps earlier, before the order existed (the previous cycle's unwind), carried in LSTM memory.
- D, ACCUM AAPL: P = 0.000 on a same-side buy while its own bid rests; the same trade flipped to the other side gives P = 1.000.
- Paper takeaway: the Watchdog's decision is the side-specific conjunction "trade opposite your own resting order", read at the current step. Its one observed failure mode is a recent opposite trade made just before an order is placed. That is a candidate for future work (e.g. order-relative trade timing).
- Checkpoint: CP-6 done.

### U-028 — CP-2: Spoofer training curves and behaviour summary (2026-10-21)
- Type: result · Paper section: Results (Claim 1: emergent spoofing), Appendix (failed runs)
- `python scripts/spoofer_training_report.py` (new; adds `matplotlib` to `requirements.txt`) → `results/spoofer_behaviour.md` and three figures in `results/figures/`, all from the committed training logs. No retraining.
- Training curves: SPOOFER-04 INTC PnL rises from about −$3.4k to +$14.5k per episode over 600k steps. Its trades against its own resting spoof rise from about 30 to about 840 per 1,000 actions. SPOOFER-02 MSFT rises to about +$2.9k and about 290. AAPL and AMZN converge near $0, with trades against their own spoof falling to about 0: they learn that spoofing does not pay there. GOOG is U-shaped, ending near +$40 and about 60.
- Failed runs figure: the free-accumulation exploit (v1) and the alternating-swing exploit (v2) both plateau at about $250k per episode, against about $14k for the final genuine run. This supports the plan's "initial failure → diagnosis → redesign" narrative.
- Behaviour, 20 episodes each, sampled actions, seed 950,000:
  - SPOOFER-04: 95% of large orders manipulative; resting time median 130 market events (p90 247); 92% cancelled; first opposite trade 10 events after placement; 171 trades against its own spoof per episode.
  - SPOOFER-02: 71% manipulative; median rest 40 events; 97% cancelled; 61 trades against its own spoof per episode.
  - SPOOFER-03: 29% manipulative; median rest 20 events; 14 trades against its own spoof per episode.
  - SPOOFER-01 / -05: 53 / 49 large orders per episode, 0.5% / 0% manipulative, median rest 10 events.
  - Realised spoof size is about 10× touch depth for all five (fixed by design).
- Checkpoint: CP-2 done.

### U-029 — CP-4: rule on real SPY order flow (2026-10-21)
- Type: result · Paper section: Results (Claim 3, real data), Limitations
- Downloaded the SPY sample (`python data/download.py --spy --tickers`; 30 levels, 09:30–10:30, 942 MB). New `scripts/real_flow_rule.py` → `results/real_flow_rule.md`, both rule settings (N = 2 × depth; M = 200 as in `basic_eval`, and M = 1000 tuned on the train split), same 30,000-event warmup.
- Consistency check: the five-stock M = 200 rows equal `results/basic_eval.json` exactly.
- SPY: 1,122,693 events after warmup, 557,100 new orders, 16 large orders. Flags: 3 at M = 200 (0.03 per 10k events), 6 at M = 1000 (0.05 per 10k).
- Context, flags per 10k events at M = 1000: MSFT 0.08, INTC 0.40, GOOG 18.15, AAPL 37.47, AMZN 106.86. On the deep, 1-tick books (SPY, MSFT, INTC) almost no real order reaches 2× the trailing touch depth. On the wide-spread stocks the rule flags 68–88% of large orders.
- Limits:
  - Real data has no labels, so these are flag rates, not false-positive rates.
  - The Watchdog cannot run on real flow, because LOBSTER has no participant identities. The plan's "real clean-data false-positive test" is therefore possible only for the rule.
  - SPY covers one hour.
- Checkpoint: CP-4 done.
