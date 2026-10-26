# RL Finance Watchdog — 10/10 Execution Plan

> **Working log — historical record, not current results.** Entries below were written as the work happened and are deliberately left unedited, including claims later superseded. For the project's current position see [docs/FINAL_AUDIT.md](docs/FINAL_AUDIT.md) and [README.md](README.md); in particular, claim 2 (generalisation) was recorded here as *not shown* and is now shown in simulation for the CP-7 detector, bounded by the label-ceiling caveat.

*Companion to `project-design-consolidated.md`. That doc explains what the system is and why. This one is the actionable roadmap to build it and close every gap identified in the rating discussion.*

---

## The Research Question (lead with this, everywhere)

> **Can reinforcement learning learn to detect previously unseen order-book manipulation strategies while maintaining a low false-positive rate on legitimate market activity?**

Every phase below exists to produce evidence for this one sentence. "Previously unseen" is answered by Phases 2, 4, 6 (held-out Spoofers, baseline comparison, scripted/real-case attacks); "low false-positive rate" is answered by Phase 3's reward design and the real-clean-data test in Phase 7.

## What "10/10" actually means here

Not "more features." A 10/10 outcome is one where, at the end, you can put these five sentences in front of a skeptical reader and back every one with a number or a plot, not an argument:

1. "The Spoofer discovered spoofing on its own" — shown via training curves + behavior analysis, not asserted.
2. "The Watchdog generalizes past the specific strategies it trained on" — shown via the held-out/scripted-attack gap.
3. "The Watchdog is actually better than what exists today" — shown via a head-to-head beat of a rule-based baseline.
4. "This isn't just simulation-specific" — shown via the real-case similarity comparison and the real-clean-data false-positive test.
5. "This is honest about its limits" — shown via an explicit, undodged limitations section.

Everything below is organized to produce evidence for those five sentences, in the order that de-risks the project fastest.

---

## Phase 0 — Data & Grounding *(done)*

- LOBSTER sample data inspected: single day (2012-06-21), 10-level message + orderbook files confirmed available for AAPL, AMZN, GOOG, INTC, MSFT; SPY confirmed only at 30/50-level depth.
- Cross-stock training decision locked (design doc §3a): AAPL/MSFT/GOOG/INTC for the training pool, AMZN held out, SPY reserved for the false-positive test.
- Legal grounding fixed: CEA §4c(a)(5)(C) (Dodd-Frank §747) — intent-to-cancel-before-execution, no pattern requirement, context matters.
- **Output:** none further needed here — carry forward.

---

## Phase 1 — Environment + Price-Impact Model + Cross-Stock Normalization

**Build:** `LimitOrderBookEnv` (Gymnasium) replaying LOBSTER data, parameterized by ticker (AAPL, AMZN, GOOG, INTC, MSFT, SPY), 43-dim observation, 6-action space (do nothing / market buy / market sell / spoof buy / spoof sell / cancel spoofs), as already scoped. Add the price-impact function: a chosen large visible order shifts `Mid_Price` used in the reward, proportional to its size relative to average depth at that level (start with a simple linear form; note square-root/Kyle's-lambda-style as a documented alternative you considered).

**Locked: cross-stock scale normalization.** Because training is now split across AAPL/MSFT/GOOG/INTC (§3a of the design doc), raw dollar prices and raw share counts cannot be used directly — a $500 move means something completely different for a $585 stock (AAPL) than a $25 stock (INTC). Observations must express price as an offset from mid (ticks or basis points) and size as a multiple of that stock's own trailing average size at that level. Build and unit-test this normalization layer before training any Spoofer variant — retrofitting it after Phase 2 would mean redoing that phase.

**Success criteria:** unit test showing that injecting a large synthetic order measurably moves the simulated mid-price used for reward, that removing the impact model collapses this to zero (proves the impact model, not an accident, is what creates the learnable signal), and that the same normalized observation encoding produces comparable-scale values across at least two stocks with different price levels (e.g., AAPL vs. INTC).

**Risk:** impact function magnitude picked arbitrarily. **Mitigation:** calibrate the coefficient against the real bid/ask size distribution you already pulled from LOBSTER (e.g., set it so a spoof at the 95th percentile of real order sizes produces a price move roughly on the order of the day's median one-tick move) — gives you a paragraph of justification instead of "we picked a number."

---

## Phase 2 — Spoofer Population Training

**Build:** 4 training-pool Spoofer variants, each on a different stock (PPO, varying seed + reward shaping): SPOOFER-01/AAPL (high inventory penalty), SPOOFER-02/MSFT (low inventory penalty), SPOOFER-03/GOOG (bid-side-only action constraint, seed C), SPOOFER-04/INTC (symmetric actions, seed D) — plus SPOOFER-05/**AMZN** as the held-out variant, trained identically but never exposed to the Watchdog, plus 1 hand-scripted rule-based attacker (layering-style, inspired by the *Coscia* pattern — multiple smaller staggered orders rather than one giant order, run on whichever stock is convenient since it isn't RL-trained). SPY is not used in this phase at all — it's reserved for Phase 7.

**Success criteria:** for each trained variant, a plot of PnL over training iterations showing convergence, plus a short behavioral description (spoof size used, hold duration, cancel timing) confirming it's actually doing something recognizable as spoof-and-cancel rather than a degenerate exploit (e.g., infinite inventory accumulation with no cancellation, which would mean the inventory penalty is miscalibrated).

**Risk (this is where projects usually break):** PPO fails to converge, or converges to a reward-hacking exploit that isn't spoofing at all. **Mitigation:** budget real iteration time for this phase — try one variant fully before parallelizing to all 4-6.

**Do not hide a failed run — report it.** If PPO initially learns something degenerate (e.g., accumulating inventory without ever cancelling, or spoofing so large it gets run over every time) and you fix it by reshaping the reward, that sequence — *initial failure → diagnosis → reward redesign → genuine spoof/cancel behavior* — is excellent research evidence, not something to edit out of the final report. It directly demonstrates the "prove convergence, don't just plan for it" requirement (addition #2) and is more convincing to a reviewer than a suspiciously clean first-try result would be.

---

## Phase 3 — Watchdog Training

**Build:** LSTM-PPO Watchdog, 20-step window over the normalized 43-dim state, trained with episodes sampling a random Spoofer **and its associated stock** from the **frozen** Phase 2 training pool (AAPL/MSFT/GOOG/INTC) — freeze the pool before this phase, don't co-train live; simultaneous co-evolution is documented as future work, not attempted here. Because observations are scale-normalized (Phase 1), the same Watchdog policy can train across all four stocks without separate models.

**Success criteria:** detection precision/recall on in-distribution episodes (pool members), reported with a confusion matrix, not just a single accuracy number (false-positive rate matters as much as recall here, per the CFTC's own "good-faith cancellation" carve-out).

---

## Phase 4 — Rule-Based Baseline *(addition #1 — highest priority, THE headline result)*

**Build:** one simple, explainable detector: flag when a visible order exceeds N× the average size at that book level over a trailing window AND is cancelled within M ticks without executing. Tune N and M on the same training-pool data the Watchdog trained on (fair comparison — baseline gets to "see" the same distribution).

**Success criteria:** run this baseline through the *exact same* evaluation matrix as the Watchdog (Phase 7) and report a side-by-side table — precision, recall, F1, false-positive rate, and detection latency, broken out per bucket (in-distribution / held-out RL / scripted attack). The target result to aim for: comparable performance in-distribution, Watchdog meaningfully ahead on held-out and scripted-attack buckets. **That gap is not a supporting detail — it is the single result the entire report should be structured around**, since it's the only piece of evidence that actually proves "adaptive beats static," which is the project's central claim. Everything else in this plan (synthetic market, real-case comparison, interpretability, multi-seed rigor) exists to make this one table more trustworthy, not to replace it.

---

## Phase 5 — Synthetic Market (sim-to-sim robustness)

**Build:** small agent-based market — a noise trader (random orders near mid) + a handful of imbalance-following reactive agents, calibrated (order-size and arrival-rate distributions) against real LOBSTER statistics.

**Updated calibration source (locked, following the cross-stock decision in §3a):** calibrate against **pooled statistics across all four training stocks (AAPL, MSFT, GOOG, INTC)**, not AAPL alone — otherwise the synthetic market would only be representative of the one regime the Watchdog trained on least specifically. Because the Spoofer and Watchdog both operate on scale-normalized observations (Phase 1), the synthetic market itself doesn't need to be rebuilt per stock or "know" which stock it represents — it only needs its background agents' relative behavior (arrival burstiness, order size as a multiple of local average) to reflect a realistic blend of the training pool rather than one stock's specific quirks. A single pooled-calibration market is enough to meet the plan; a stretch upgrade, if time allows, is building two variants — one leaning toward the more liquid stocks (AAPL/MSFT) and one toward the thinner one (INTC) — to check the robustness result holds across both liquidity regimes, not just on average.

Drop the already-trained, frozen Spoofer pool and Watchdog in with no further training.

**Success criteria:** Watchdog detection performance reported in this environment alongside the historical-replay numbers — the delta between them is your sim-to-sim generalization result.

---

## Phase 6 — Real-Case Similarity Analysis *(addition #3 — second priority)*

**Build:** pull the public trade-level details from the *Coscia* (and/or *Sarao*) case filings — order sizes, layering structure, cancel timing, as reported in court records and CFTC/DOJ press releases and exhibits. Compute the same descriptive statistics (order-size distribution relative to book depth, cancel latency) for your trained Spoofer's emergent behavior, and present them side by side — a table or overlaid histogram, not just a verbal claim of resemblance.

**Success criteria:** a quantitative comparison a reader can evaluate themselves, explicitly flagged as "structural similarity to a documented real case," not "validated against real data" (keep the claim precisely scoped, per the honesty requirement in the design doc).

---

## Phase 7 — Full Evaluation Matrix + Real Clean-Data False-Positive Test

**Run** all of: in-distribution (AAPL/MSFT/GOOG/INTC), held-out RL Spoofer (SPOOFER-05 on AMZN — unseen strategy *and* unseen stock), scripted attack, synthetic market, baseline-vs-Watchdog comparison, AND the real-world false-positive check (**SPY**, zero injected spoofing, never touched during training).

**Success criteria:** one consolidated results table covering every bucket for both the Watchdog and the baseline, **plus a second, plainly-stated economic result**: Spoofer PnL with no Watchdog vs. Spoofer PnL with the Watchdog active, reported as a suppression percentage. This PnL-suppression number is the one to quote first in an abstract or executive summary — it's legible to a non-technical reader in a way precision/recall never will be, and it directly answers "did this actually stop the manipulation from paying off," which is the real-world stakes of the whole project.

---

## Phase 8 — Interpretability *(addition #4)*

**Build:** extract LSTM attention weights (or, if attention isn't part of the architecture, a feature-ablation pass — zero out each of the 43 features one at a time and measure the drop in detection confidence) for a handful of true-positive flags. Present 2-3 concrete examples: "the Watchdog flagged this episode primarily because of [feature], consistent with the size-spike-then-cancel pattern."

**Success criteria:** at least one concrete, human-readable explanation of a real flag, directly answering the "black box" deployment objection from the design doc.

---

## Phase 9 — Statistical Rigor Pass *(addition #5)*

**Build:** re-run Phase 3 (Watchdog training) and the Phase 7 evaluation across 3 random seeds. Report mean ± confidence interval for the key numbers (detection rate gap between in-distribution and held-out, PnL suppression), not single-run point estimates.

**Success criteria:** every headline number in the final report has an uncertainty range attached.

---

## Phase 10 — Write-Up

Assemble everything into the final report, in this order: research question → legal grounding → system → **headline results first** (baseline-vs-Watchdog table + PnL suppression, per Phase 4/7) → the rest of the evaluation matrix (synthetic market, real-case similarity) → interpretability examples → explicit, undodged limitations section (domain gap, no labeled real data, false-positive cost, latency/scale — as drafted in the design doc's §6) → future work.

**Use this sentence, close to verbatim, to describe what the project accomplished:**

> "This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases."

**Never use this sentence:**

> "Our system can detect real-world spoofing."

No labeled real-world spoofing dataset exists to support that claim — see design doc §6. Getting this line right is the cheapest, highest-leverage thing in the entire plan: it costs nothing to write, and it's the difference between a reviewer trusting everything else in the report or discounting it.

---

## If Time Is Tight — Priority Order

Core system (Phases 0-3) is non-negotiable — it's the thing everything else evaluates. Beyond that, if you have to cut, cut in this order (highest-value-per-hour first, trim from the bottom):

1. **Phase 4 — Rule-based baseline.** Non-negotiable. Without this there is no proof of the central thesis at all.
2. **Phase 6 — Real-case (Coscia/Sarao) similarity comparison.** Cheapest available way to strengthen the real-world connection; do this before anything fancier.
3. **Phase 5 — Synthetic market.** Valuable (it's the one place the market mechanism itself changes, not just the data), but can be shallowed to a minimal 2-3-agent version if time is short.
4. **Phase 8 — Interpretability.** Real value for the deployment-credibility argument, but the project's central claim survives without it.
5. **Phase 9 — Multi-seed statistical rigor.** Trim to a single seed with a clearly stated caveat before cutting anything above this line.

Never cut Phase 4. It's the one piece of evidence that turns "we built two clever agents" into "we proved adaptive detection beats static rules."

---

## Risk Register (what's most likely to actually go wrong)

| Risk | Phase | Fallback |
|---|---|---|
| PPO doesn't converge / degenerates | 2, 3 | Report the failure + fix as a finding (this is legitimate content, not a hole) |
| Impact model coefficient looks arbitrary | 1 | Calibrate against real LOBSTER order-size percentiles, document the choice explicitly |
| Cross-stock normalization has a bug (e.g. wrong average-size window) that silently corrupts training on one or more stocks | 1, 2 | Unit-test the normalization layer independently, per stock, before starting any Spoofer training — this is the one bug class that wastes an entire Phase 2 run if caught late |
| Baseline detector is too weak/too strong (trivial comparison) | 4 | Tune N/M on held-out data too, so the comparison isn't rigged either direction |
| Real case data (Coscia/Sarao) hard to extract cleanly | 6 | Use whatever level of detail is publicly available (even order-of-magnitude comparisons on size/timing count) and state the limitation |
| Not enough time for all 10 phases | any | Use the minimum path above; document what was cut as explicit future work rather than silently dropping it |

---

*Companion execution plan — 2026-09-01.*
