# An Adversarial Reinforcement Learning System for the Detection of Limit Order Book Spoofing

**Project report.** All figures and numbers in this report are generated from the committed code and results in
this repository (`results/`, `results/figures/`); none are illustrative. Every evaluation script has been
re-run in a clean clone and reproduces its reported values exactly (2,178 numeric values compared, 0 differences).

---

## Abstract

Market-surveillance systems in production are predominantly static rule sets. This project tests whether a
reinforcement-learning detector, trained adversarially against RL agents that discover manipulation on their
own, generalises better than such a rule. Using LOBSTER limit-order-book data for five NASDAQ stocks, a
population of PPO "Spoofer" agents was trained with no instruction to spoof — only to maximise profit — and a
recurrent "Watchdog" was trained to flag manipulation from order-flow observations, with no spoofing rule
supplied. Both were compared against a rule-based baseline (large order cancelled quickly) on identical data.

Spoofing emerged in agents trading the two narrow-spread stocks: profit of $14,368 ± 1,135 (INTC) and
$11,431 ± 16,844 (MSFT) per episode across three training seeds, which becomes a loss of $7–8k when the
price-impact model is disabled, confirming the profit comes from the manipulation itself. The Watchdog
achieved an order-level F1 of 0.753 ± 0.058 against the rule's 0.352, and a false-positive rate on legitimate
orders of 0.031 ± 0.067 against 0.823 — roughly 26× fewer false accusations. However, on two **pre-registered**
held-out attacks it recalled only 0.22 and 0.30 of manipulative orders where the rule recalled 0.89 and 1.00.

The central thesis — that an adaptive detector generalises better than a static rule — is therefore **not
supported for recall**, and **supported for precision and false-positive cost**. A feature ablation explains
why: detection rests almost entirely on observing the agent trade against its own resting order, so sparse
manipulation evades it.

The held-out failure was later diagnosed and fixed. The hindsight label marked every step an order rested,
including steps before any opposite trade, so training rewarded predicting *who would spoof* over detecting *the
act*. With a causal label, a myopic objective and training coverage of varied act timing, the CP-7 Watchdog
reaches order-level recall **0.98–0.99 at a legitimate false-positive rate of 0.001–0.012 across four held-out
attack families**, including a pre-registered test it was the frozen verdict model for.

A subsequent audit (§6.1) places a bound on what that is worth. The manipulation label is
`opposite_trades > 0`, and two of the detector's own input features state the participant's trade direction
and the side of its resting orders, so a **zero-parameter comparison of those two features scores an
order-level F1 of 0.997 in-distribution and 0.995 on the held-out scripted attacker, at a false-positive rate
of 0.000** — and gradient boosting fitted on the train split reaches 0.96 and 0.99. The detection task as
posed is therefore close to a tautology, the learned detector solves it worse than a trivial rule, and the
genuine difficulty of spoofing surveillance — attributing orders and trades to an account — is assumed away by
the environment. The CP-7 Watchdog therefore reached a ceiling that a zero-parameter rule
already occupied, and a supervised LSTM reaches it too, in minutes. The detector arc is a real and instructive
result about **labelling**, not evidence that reinforcement learning solves manipulation detection; the
contribution that stands on its own is the emergence of profitable manipulation on the attacker side. A further sim-to-sim test found that the Spoofers' profitability does not
survive a market where prices emerge from order matching rather than from an impact model, which bounds the
scope of the first result.

---

## 1. Research question

> Can a reinforcement learning agent (the "Watchdog") learn to detect *previously unseen* order-book
> manipulation strategies — specifically spoofing — while maintaining a low false-positive rate on legitimate
> market activity, without being given explicit rules about what spoofing looks like?

This decomposes into three claims, each requiring evidence rather than assumption:

1. RL learns meaningful manipulation (not a reward-hacking artefact).
2. The Watchdog generalises to attacks it has not seen. *(Not supported for the original Watchdog; supported
   in simulation for CP-7 — but §6.1 shows the task was not a hard test of it.)*
3. It does not simply flag legitimate activity.

**Scope statement, used verbatim throughout:** this project demonstrates the feasibility of an RL-based
surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of
the emergently-learned manipulation strategy to documented real-world manipulation cases. No labelled
real-world spoofing dataset exists, so **no claim of real-world detection capability is made**.

## 2. Background and legal grounding

Spoofing — "bidding or offering with the intent to cancel the bid or offer before execution" — is prohibited
under CEA §4c(a)(5)(C), added by §747 of the Dodd-Frank Act. CFTC guidance establishes that intent matters,
that a single occurrence can violate, and that context must be evaluated. Two design choices follow directly:
the Watchdog observes a window of history through a recurrent policy rather than classifying isolated
snapshots, and it is able to flag a single occurrence rather than requiring a repeated pattern.

Two enforcement actions provide the structural reference points used in §7: the CFTC order against Panther
Energy Trading and Michael Coscia (Docket 13-26, 2013) with the subsequent Seventh Circuit opinion, and the
CFTC's 2015 action against Navinder Sarao.

## 3. Method

### 3.1 Data

LOBSTER sample message and order-book files for 2012-06-21, ten levels deep: AAPL, MSFT, GOOG, INTC, AMZN
(400k–670k events each). The two stock types differ sharply, which matters for every result below: AAPL and
GOOG trade near $570–585 with 15–28 tick spreads and ~140 shares at the touch, while MSFT and INTC trade near
$27–30 with 1-tick spreads and ~13,500 shares at the touch.

### 3.2 Simulation environment

Historical replay cannot react to an injected order, so a price-impact layer is added. Its coefficient is
**fitted per stock from order-flow imbalance** (Cont, Kukanov & Stoikov, 2014) rather than assumed: regressing
mid-price change on signed order flow gives λ = 0.046 (AAPL) to 0.255 (MSFT), explaining 33–73% of variance.
A spoof's impact builds with the order's age following the measured response at 10/50/200/1000 events, and its
favourable price shift applies to at most its own size in opposite-side volume.

Cross-stock comparability required scale-free features: prices as offsets from mid in units of the stock's
trailing median spread, sizes as log ratios to trailing depth. In tick units the same book shape differs ~15×
between AAPL and INTC; in spread units both yield a median best-level offset of 0.500.

Agents act once per 10 market events, choosing among: no-op, market buy, market sell, place a large order at
the touch (10× trailing depth), or cancel. Inventory is valued at the **unimpacted** historical mid, so merely
holding a position while a spoof rests earns nothing; profit requires trading at the distorted price.

### 3.3 Agents

Five PPO Spoofers (256×256 MLP), one per stock, differing in inventory penalty and action constraints. The
Watchdog is a RecurrentPPO (LSTM) policy observing 46 features — 40 book features plus six describing the
monitored participant's own order activity — and choosing at each step to flag or clear. Its reward is
asymmetric, reflecting real surveillance economics: +1 correct flag, −1 miss, **−2 false accusation**, 0
correct clear. It is never shown the rule that defines the label.

The rule-based baseline flags any order at least N× trailing depth cancelled within M events without
executing, with N and M tuned on the training split — the same data the Watchdog trained on.

### 3.4 Evaluation protocol and safeguards

Four safeguards were adopted because early results proved untrustworthy (§6):

- **Pre-registration.** Two held-out attackers (LATEBURST-ATK, LAYER-ATK) had their behaviour, evaluation
  procedure and hypothesis committed to version control *before* any of their data existed, and each test was
  run exactly once. Both went against the Watchdog and are reported as such.
- **Counterfactual decomposition.** Every claim of manipulation profit is checked by re-running the identical
  policy with price impact disabled.
- **No tuning on held-out data.** One candidate design change was evaluated on the test split only, and
  rejected on that evidence.
- **Multiple seeds.** Three training seeds for the Watchdog and for each profitable Spoofer, with confidence
  intervals.

## 4. Result 1 — manipulation emerges, within limits

Spoofing was never rewarded directly; only profit was. In the two narrow-spread stocks the agents discovered
spoof-and-trade behaviour independently.

![Spoofer training](results/figures/fig1_spoofer_training.png)

**Figure 1.** (a) At one decision per market event the agent converged to never trading: the ~50-event delay
between placing a spoof and profiting from it was 50 decisions away, and random early trading cost ~$13.6k per
episode, so inaction dominated. Acting once per 10 events solved the credit-assignment problem. (b) The
spoofing signature — trading against one's own resting order — emerging over training.

![Seed robustness](results/figures/fig5_seed_robustness.png)

**Figure 2.** All three seeds for both stocks learned to spoof. Switching off price impact turns every seed's
profit into a $7–8.6k loss, establishing that the profit *is* the manipulation and not some other artefact.
INTC is stable across seeds ($14,368 ± 1,135); MSFT varies widely ($11,431 ± 16,844).

Spoofing did **not** emerge on AAPL, GOOG or AMZN. Under the calibrated impact, a spoof there moves the price
0.37–0.62 spreads — less than the half-spread cost of crossing — so it cannot pay. This is a property of those
markets under this model, not a training failure, but it does thin the agent population and leaves the held-out
RL test (AMZN) with no manipulation to detect.

## 5. Results 2 and 3 — the surveillance trade-off

![Detection comparison](results/figures/fig3_detection_comparison.png)

**Figure 3.** Watchdog versus rule across every evaluation bucket.

| Measure | Watchdog | Rule | Source |
|---|---|---|---|
| In-distribution order F1 | **0.753 ± 0.058** | 0.352 | `multiseed.md` |
| False positives, legitimate orders | **0.031 ± 0.067** | 0.823 | `multiseed.md` |
| False positives, held-out stock | **0.004 ± 0.018** | 0.972 | `multiseed.md` |
| Recall, pre-registered unseen attacker | 0.217 ± 0.156 | **0.885** | `lateburst_test.md` |
| Recall, pre-registered layering attack | 0.30 | **1.00** | `layering_test.md` |
| Detection delay (median) | **0 steps** | 8 steps | `watchdog_eval.md` |

Two findings stand out.

**Claim 3 is supported decisively.** The Watchdog wrongly flags 3–7% of legitimate large orders where the rule
flags 82–97%. This is the difference between a usable surveillance system and an unusable one: at a 0.82
false-positive rate, every genuine alert arrives buried in false accusations against legitimate traders. The
rule's weakness is structural — large orders cancelled quickly are ordinary market-making behaviour. Its flag
rate on real, unlabelled order flow varies by a factor of 3,400 across stocks (0.03 per 10k events on MSFT,
101.9 on AMZN), confirming that one static threshold cannot serve different market structures.

**Claim 2 is not supported.** On both pre-registered attacks the rule recalls far more. The Watchdog also flags
*while the order still rests* (median delay 0 steps, against 8 for the rule, which can only fire at the
cancellation) — operationally valuable, since intervention is possible before the manipulation completes.

![PnL suppression](results/figures/fig6_pnl_suppression.png)

**Figure 4.** When the detector acts inside the episode — flagging triggers cancellation of the trader's
resting orders and a 200-event block — the Watchdog removes 83% of SPOOFER-04's profit against the rule's 55%,
and turns SPOOFER-02's profit into a loss. It intervenes against legitimate traders ~12× less often. Against
manipulation it recognises, the adaptive detector is both more effective and less disruptive.

## 6. Why the Watchdog fails, and what failed along the way

### 6.1 The detection task is close to a tautology

The ablation below shows the Watchdog relies almost entirely on one feature. A direct measurement explains
why that is inevitable: **the label is present in the observation.**

The manipulation label is `opposite_trades > 0` — the order's owner traded on the opposite side while the
order rested. Observation feature 44 is the participant's trade direction this step and feature 45 is the side
its resting orders sit on, so comparing their signs reproduces the label's definition. Scoring that comparison
as a detector, with **no parameters and no fitting at all**:

| Group | Manipulative / orders | Precision | Recall | Order F1 |
|---|---|---|---|---|
| In-distribution (test split) | 1,543 / 7,243 | 0.999 | 0.994 | **0.997** |
| Held-out scripted attacker | 2,420 / 2,580 | 1.000 | 0.990 | **0.995** |
| Legitimate only (HONEST + FLICKER) | 0 / 2,688 | — | — | FPR 0.000 |

"A resting order is present" alone gives order F1 0.362 at FPR 1.000, so it is specifically the two-feature
comparison that carries the label, not the presence of a large order. Supervised detectors confirm it: fitted
on the train split with thresholds chosen on train orders only, gradient boosting reaches order F1 0.96
in-distribution and 0.99 on both scripted-attacker buckets, against the Watchdog's 0.77, 0.38 and 0.26.
(`scripts/label_observability.py`, `scripts/supervised_baseline.py`.)

Three things follow, and they bound every detector result in this report:

1. **The Watchdog is not discovering manipulation.** It is recovering, imperfectly, a definition it was handed
   in its own input. Its F1 of 0.753 against the rule's 0.352 is a real difference between two restricted
   detectors — the rule may only fire at cancel time — but it is not evidence that learning was required.
2. **The generalisation failure is a property of the learner, not of the features.** A zero-parameter
   signature transfers to the unseen scripted attacker unchanged. The Watchdog's collapse there therefore
   cannot be attributed to the observation lacking the signal, and the hypothesis in §1 claim 2 is not merely
   unsupported — it was tested on a task where the information was always available.
3. **The real simplification is perfect attribution.** The environment tells surveillance which orders and
   trades belong to the monitored participant. A regulator or exchange with account-level data has that;
   anonymised feeds such as LOBSTER do not. Establishing attribution — not recognising the pattern once
   attribution is given — is the hard part of spoofing detection, and this project assumes it away.

This was found by audit after the main results were produced, and is reported here rather than reframed.

### 6.1b The fix: a causal label, and what it proves

The original Watchdog's failure was a labelling artefact, not a learning failure. The hindsight label marked
**every** step an order rested, so the objective rewarded identifying the *actor* rather than the *act*. The
diagnostic is unambiguous: the original Watchdog flagged 99% of SPOOFER-04's orders before any opposite trade
occurred, and only 13.7% of SPOOFER-03's opposite-trade steps.

Three changes, all selected on test splits and never on held-out data: a **causal label** (positives begin at
the first opposite trade), a **myopic γ = 0 objective**, and **training coverage of varied act timing**.

| Attack family | Original Watchdog | CP-7 `cov` | Rule | Label ceiling |
|---|---|---|---|---|
| SCRIPTED-ATK (train stocks) | 0.257 ± 0.199 | **0.991 ± 0.001** | 0.938 | ~0.99 |
| LATEBURST-ATK | 0.217 ± 0.156 | **0.984 ± 0.001** | 0.885 | 0.983 |
| DRIP-ATK | 0.067 ± 0.115 | **0.989 ± 0.002** | 0.744 | 0.989 |
| REVERSAL-ATK | 0.139 ± 0.129 | **0.983 ± 0.005** | 0.964 | 0.984 |

Order-level recall, 3 seeds, mean ± 95% Student-t CI. Only LATEBURST (for the original Watchdog) and
REVERSAL-ATK (for `cov`) were pre-registered verdicts; the others are post-hoc descriptions.

**What this establishes:** a labelling choice can silently redefine a task, and finding that was worth more than
any hyperparameter search. **What it does not establish:** that RL was required. With γ = 0 the Watchdog's
training is cost-weighted classification by policy gradient, and a supervised LSTM with the same labels matches
it on every dataset while training in minutes. Logistic regression fails (recall 0.20–0.47), which shows the
task needs the interaction between trade side and resting side — a two-feature interaction, which is §6.1's
point restated.

### 6.2 Interpretability: what the Watchdog actually learned

A feature-group ablation on the test split isolates the cause. Removing the single feature "did this
participant trade, and in which direction" collapses order-level F1 from 0.772 to **0.113** and step recall to
0.018. Removing all six participant features leaves nothing (F1 0.001), while removing all forty book features
*improves* F1 slightly.

The Watchdog therefore learned one specific signature: an agent trading repeatedly against its own resting
order. The dense MSFT/INTC Spoofers do this on almost every step. Sparse manipulators — a wait, then a few
trades — do not, and slip through. This is visible inside training too: SPOOFER-03 (GOOG) spoofs on 21% of its
test steps and is flagged on 1.8%.

### 6.3 Environment exploits found and closed

Headline profit was repeatedly misleading. Four exploits were caught only by decomposing where PnL came from:

| Exploit | Symptom | Fix |
|---|---|---|
| Free accumulation | Bought ~20× touch depth below the real bid; +$253k/episode | Charge the agent's own market orders price impact |
| Alternating-side swing | ±40-lot swings cost $53 while the spoof shift applied to unlimited volume; ~$240k/episode | Impact builds with order age; favourable shift capped at the spoof's size |
| Early termination | All 31 of 31 early endings occurred on a run-over step — an escape from losses | Lot scaled to depth; no early termination |
| Thin opening book | INTC depth 0.08× its median at event 15,000 | 30,000-event warmup |

### 6.4 Training collapses

![Watchdog training](results/figures/fig2_watchdog_training.png)

**Figure 5.** The Watchdog initially collapsed to never flagging, at every entropy coefficient tried (0, 0.01,
0.03, 0.1). Early random flags land mostly on legitimate steps at −2 each, so silence (−0.13/step) dominates
before the features are learned. The cause was input scale: book features span ±20 while the informative
participant features span 0–5. Standardising observations fixed it; with raw inputs the collapse recurred even
at a milder −1 false-positive cost, confirming scale rather than reward asymmetry as the cause.

## 7. Structural comparison with documented cases

| Dimension | Coscia (court record) | SPOOFER-04 | Emergent or fixed by design |
|---|---|---|---|
| Large orders filled | 0.08% (CME), 0.5% (ICE) | 0.4% | Emergent |
| Large orders resting > 1 s | 0.57% | 52% | Emergent |
| Reversal after a cycle | "designed to promptly operate in reverse" | 74% of consecutive orders flip side | Emergent |
| Large vs small order size | 5 vs 50+ contracts | ~17–20× the trade lot | Fixed by the environment |

The learned strategy matches the documented pattern where the agent has freedom: large orders on one side
while trading the other, and almost never allowing the large order to fill. It differs in speed — the
calibrated impact ramp rewards patience, whereas Coscia's cycle completed in about two-thirds of a second —
and it cannot layer, because the environment permits one large order per side.

## 8. Robustness: does any of this survive a different market?

The replay environment imposes price impact by formula. As a sim-to-sim check, an agent-based market was built
in which prices emerge purely from order matching between noise traders and imbalance-following traders, and
calibrated against real MSFT/INTC statistics (spread, mid-change frequency, 2,000-event range, touch depth,
event mix) — never against spoof profitability.

![Spoof impact](results/figures/fig4_spoof_impact.png)

**Figure 6.** A spoof does move the emergent price, with the same build-up shape and no formula involved — but
4.5–9× less than the replay model assumes, across the range of follower populations consistent with real price
dynamics.

Consequently **every manipulator loses money in the synthetic market**, including both RL Spoofers, and no
manipulation gain is clearly distinguishable from zero. Detection partially transfers: the Watchdog holds
precision 0.88 and a 0.11 false-positive rate against the rule's 0.92, but recall falls to 0.48.

This is the most important limitation in the project. Claim 1 holds under the replay impact model and not
beyond it. Which model is closer to reality is unresolved: the synthetic market's only reactive channel is
imbalance-following, with no market makers repricing quotes, so it is plausibly too inert — but that is a
hypothesis, not a defence.

## 9. Discussion

The detector comparison is a trade-off rather than a victory. The static rule catches nearly everything,
including nearly everything innocent; the adaptive detector is precise, quiet, early, and blind to
manipulation structured differently from what it has seen. Neither could be deployed as measured here.

That comparison, however, is not the most important thing the project established about its own detector.
Because the label is recoverable from two of the detector's input features (§6.1), neither detector was facing
a hard problem, and the right reading of the 0.753-versus-0.352 gap is that two differently handicapped
detectors sit well below a zero-parameter signature at 0.997. **The honest conclusion on the detector side is
negative: the RL formulation is not justified for this task, and the task is not a test of the research
question it was built to answer.** A supervised LSTM matches the CP-7 Watchdog on every held-out dataset in
minutes, and on the original labels gradient boosting dominates the original recurrent policy everywhere.

This does not make the detector work valueless — it relocates its value. The diagnosis in §6.1b is a genuine
methodological result: the project caught a labelling artefact that had quietly turned a detection task into an
actor-identification task, and demonstrated the fix across four independently pre-registered attack families.
That is a transferable lesson. The 0.98 recall is not.

The ablation points to why the Watchdog in particular does poorly: it keyed on a single behavioural signature
because the training population exhibited one dominant style, and because that signature is the label. The
instructive redesign is therefore not "train against more styles" but **remove the label from the
observation** — require the detector to infer intent from order-flow structure without being told which
trades belong to the resting order's owner. That is a different and genuinely hard problem, and it is the one
real surveillance faces.

Two methodological results are worth stating independently of the detector. First, **headline profit is not
evidence of learned manipulation** — four separate exploits produced large, plausible-looking profits, and each
was caught only by decomposing the PnL and running an impact-free counterfactual. Second, **pre-registration
changed the conclusions**: the earlier held-out attacker had been inspected during analysis and was no longer a
clean test, and both pre-registered replacements went against the hypothesis.

## 10. Limitations

- The train/test splits originally drew episodes from the whole trading day, so 33 of 50 test episodes
  overlapped a training episode. Fixed (disjoint windows per split); the measured cost was 0.031 order F1
  in-distribution, below the ±0.058 seed spread, and nothing on the held-out buckets
  (`results/temporal_split/comparison.md`).
- **The detection task is close to a tautology (§6.1).** The label is readable from two observation features, a
  zero-parameter signature scores 0.997 in-distribution and 0.995 on a held-out attacker, and gradient boosting
  beats the RL detector everywhere. No claim that this system "detects spoofing" in a meaningful sense is
  supported, and reinforcement learning is not justified for the detector.
- **The environment grants surveillance perfect attribution** of orders and trades to one monitored
  participant. Establishing that attribution is the hard part of real spoofing surveillance and is assumed away.
- Profitable manipulation is established only under the replay impact model (§8).
- The detector misses sparse manipulation — confirmed twice under pre-registration, but see §6.1 for why this
  is a statement about the learner rather than about detectability.
- LOBSTER carries no participant identities, so the Watchdog cannot be evaluated on real order flow; only the
  rule can, and only as a flag *rate* with no labels.
- One trading day, five stocks, single-day calibration.
- The manipulation label is an economic proxy for intent (trading the opposite side while an order rests), not
  a legal determination.
- Layering can be scripted but not learned, as the environment allows one large order per side.
- Each non-profitable Spoofer remains single-seed.

## 11. Future work

1. **Remove the label from the observation.** Withhold per-participant attribution and require the detector to
   infer manipulation from aggregate order flow. This is the change that turns the detector half of the project
   into a real problem, and §6.1 shows why nothing else on the detector side matters until it is made.
2. Resolve whether the replay impact model is too generous or the synthetic market too inert — this decides the
   scope of claim 1.
3. Train the Watchdog against a population spanning sparse and dense manipulation styles, and re-run both
   pre-registered tests.
4. Extend the action space to multiple resting orders per side so layering can be learned rather than scripted.
5. Co-evolution (currently the Spoofer pool is frozen before Watchdog training) as a route to an adaptive
   adversary.

## 12. Reproducibility

All code, trained models, results and figures are committed. From a clean clone with the data downloaded, every
evaluation reproduces exactly: **2,178 numeric values compared across 12 scripts, 0 differences**, plus all 32
recorded dataset files bit-for-bit. The test suite is 139 tests against real market data. `HANDOFF.md`
documents the full procedure; `dashboard.html` presents the results interactively.

## References

1. Commodity Exchange Act §4c(a)(5)(C), as added by §747 of the Dodd-Frank Wall Street Reform and Consumer
   Protection Act (2010).
2. CFTC, *Antidisruptive Practices Authority* interpretive guidance, 78 Fed. Reg. 31890 (2013).
3. CFTC, *In re Panther Energy Trading LLC and Michael J. Coscia*, Docket 13-26 (22 July 2013).
4. *United States v. Coscia*, 866 F.3d 782 (7th Cir. 2017).
5. CFTC, *CFTC Charges U.K. Resident Navinder Singh Sarao … with Price Manipulation and Spoofing*, Release
   7156-15 (2015).
6. Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial
   Econometrics*, 12(1), 47–88.
7. Huang, R. & Polak, T. (2011). LOBSTER: Limit Order Book Reconstruction System. Humboldt University Berlin.
8. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
9. Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268).
