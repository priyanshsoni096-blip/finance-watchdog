# Handoff — RL Finance Watchdog

For whoever picks this up next. It assumes no prior contact with the project. `README.md` is the reference manual; this document is the orientation: what the project is, what is actually established, what is not, and what I would do next.

Repository: <https://github.com/priyanshsoni096-blip/finance-watchdog> (all work is pushed to `main`).

---

## 1. What the project is

> Can a reinforcement learning agent (the "Watchdog") learn to detect *previously unseen* order-book manipulation strategies — specifically spoofing — while maintaining a low false-positive rate on legitimate market activity, without being given explicit rules about what spoofing looks like?

Two populations of agents in a limit-order-book simulation built on real LOBSTER data (NASDAQ, 2012-06-21, five stocks):

- **Spoofers** (PPO) learn to trade profitably. Nobody tells them to spoof; spoofing has to emerge because it pays.
- **The Watchdog** (RecurrentPPO, LSTM) watches one trader's order activity and flags manipulation. It is never given the spoofing rule, only rewards.
- **A rule-based baseline** ("large order cancelled quickly") is the bar to beat. It stands in for the static surveillance systems exchanges actually run.

**Language rule, kept verbatim in every deliverable:**

> This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

Never write "our system can detect real-world spoofing". No labelled real-world spoofing data exists, so detection recall on real manipulation cannot be measured. That is a limitation of the field, not of this work.

Legal reference: CEA §4c(a)(5)(C) (Dodd-Frank §747).

---

## 2. Where it stands

Everything below is measured and committed under `results/`. Nothing is estimated.

| Claim | Verdict | Key evidence |
|---|---|---|
| 1. RL learns meaningful manipulation | **Partly shown** | SPOOFER-04 (INTC) +$13,208 ± 1,654 per episode, SPOOFER-02 (MSFT) +$2,437 ± 1,068. With price impact switched off the same trading loses money, so the profit is the spoof. Across 3 training seeds each, all 3 seeds learned to spoof on both stocks (INTC +$14,368 ± 1,135; MSFT +$11,431 ± 16,844). But in the synthetic market both lose money, so profitability holds only under the replay impact model. |
| 2. The Watchdog generalises to unseen attacks | **Not shown for the original Watchdog; shown in simulation for CP-7 — but against a ceiling a zero-parameter rule already occupies** | Original Watchdog: LATEBURST recall 0.24 vs rule 0.89; layering 0.30 vs 1.00. Diagnosed as a labelling artefact (§2c), fixed with a causal label, γ = 0 and varied-timing training coverage. CP-7 `cov`: recall **0.98–0.99** at legitimate FPR 0.001–0.012 across four held-out attack families, **and both originally-failed pre-registered tests now pass** — LATEBURST 0.22→0.984, LAYER 0.30→0.990 ± 0.025 in the synthetic market. But the label-definition rule scores the same by construction and a supervised LSTM matches it — see §2b. |
| 3. It does not flag legitimate activity | **Shown in simulation** | False-positive rate on legitimate orders 0.031 ± 0.067 across 3 seeds vs the rule's 0.823; 0.11 vs 0.92 in the synthetic market. Not testable on real data (see §10). |

**The detector in one line:** the original Watchdog failed on unseen attacks, the cause was the label rather
than the learner, the fix worked completely — and the ceiling it reached is occupied by a rule with no
parameters. Read §2b and §2c together; neither is the whole story on its own.

Supporting results:

- **In-distribution detection**: CP-7 `cov` order-level F1 **0.984 ± 0.001** (3 seeds) vs rule **0.352**; the
  original Watchdog was 0.753 ± 0.058. It flags while the order still rests (median delay 0 steps); the rule can only fire at the cancel (8 steps).
- **PnL suppression**: with the Watchdog acting inside episodes (flag → cancel the trader's resting orders, block new ones for 200 events), SPOOFER-04's profit falls **83%** vs the rule's 55%, while intervening on legitimate traders about 12× less often.
- **Interpretability**: detection rests almost entirely on one feature. Ablating "trade direction" drops F1 from 0.772 to **0.113**. That is also why sparse spoofing is missed.
- **Real cases**: structurally similar to Coscia in the dimensions the agents choose (large orders one side, trading the other, fills 0.4–1.4% vs his 0.08–0.5%, side alternation). Different in speed (44–52% of our spoofs rest over 1 s vs his 0.57%) and no layering in the replay environment.

---


### 2b. Read this before you quote any detector number

An audit after the main results found that **the detection task as posed is close to a tautology.** The label
is `opposite_trades > 0` — the owner traded the opposite side while the order rested — and observation feature
44 is the participant's trade direction while feature 45 is the side its resting orders sit on. Comparing their
signs *is* the label.

Measured with no parameters and no fitting (`python scripts/label_observability.py`):

| Group | Precision | Recall | Order F1 |
|---|---|---|---|
| In-distribution (test split) | 0.999 | 0.994 | **0.997** |
| Held-out scripted attacker | 1.000 | 0.990 | **0.995** |
| Legitimate only | — | — | FPR **0.000** |

Supervised baselines agree (`python scripts/supervised_baseline.py`): gradient boosting fitted on train only
reaches order F1 0.96 in-distribution and 0.99 on both scripted buckets, against the Watchdog's 0.77 / 0.38 /
0.26. **Reinforcement learning is not justified for the detector.**

What this does and does not change:

- **Does not change** the Spoofer side. Agents discovering profitable spoofing under a calibrated impact model
  is unaffected, and it is the project's real contribution.
- **Does change** how every detector number must be presented. Watchdog 0.753 vs rule 0.352 is a comparison of
  two handicapped detectors far below a trivial signature, not evidence that learning helps.
- **The genuine hard problem is attribution.** The environment hands surveillance the knowledge of which orders
  and trades belong to the monitored participant. Real surveillance must establish that first; anonymised data
  such as LOBSTER cannot. Removing per-participant attribution from the observation is the single change that
  would make the detector half of this project a real experiment, and it is the first item in future work.

Locked down by `tests/test_label_observability.py` so a future change to the label or features cannot quietly
invalidate this. Full write-up: `docs/FINAL_AUDIT.md` §C and `REPORT.md` §6.1.

Arnav reached the same conclusion independently, from the other direction: he built **LABEL-RULE**, the label's
definition written as a rule, and recorded it as *"the ceiling, perfect by construction, not a competitor."*
Two people finding this separately is the strongest evidence it is real.

### 2c. How the detector was fixed, and what the fix does and does not prove

The original Watchdog's held-out failure was **not** a learner problem. The hindsight label marked every step an
order rested, including steps before any opposite trade, so the objective rewarded predicting **who would spoof**
over detecting **the act**. The diagnostic is unambiguous (`scripts/flag_timing_diagnostic.py`): the original
Watchdog flagged 99% of SPOOFER-04's orders *before* any opposite trade, and only 13.7% of SPOOFER-03's
opposite-trade steps.

Three changes, all chosen on test splits and never on held-out data:

1. **Causal label** (`watchdog.dataset.causal_labels`) — positives begin at the first opposite trade.
2. **γ = 0**, a myopic objective. With γ = 0 the Watchdog's RL training is cost-weighted classification by
   policy gradient.
3. **Varied act timing in training** (VARIED-SPOOF, ACCUM), kept in a separate `data/watchdog_ext` so the
   original splits and every result built on them are unchanged.

Result across four held-out attack families: recall 0.22–0.30 → **0.98–0.99**, legitimate FPR 0.001–0.012
(`results/posthoc_rescore.md`). It also transfers to the **synthetic market**, a different price-formation
mechanism: layering recall 0.30 → 0.990 ± 0.025 and manipulator F1 0.62 → 0.980 ± 0.011, at a legitimate FPR
of 0.030 against the rule's 0.917 (`results/posthoc_summary.md`, 3 seeds).

**What this proves:** a labelling choice can silently redefine a task, and diagnosing that is worth more than any
amount of hyperparameter search. **What it does not prove:** that RL was needed. A supervised LSTM with the same
labels matches it everywhere and trains in minutes (`results/baselines_final.md`).

## 3. Getting productive in ~30 minutes

Windows 11, Python 3.13, CPU only. There is no GPU requirement anywhere.

```bash
pip install -r requirements.txt
```

```bash
python data/download.py
```
~600 MB of LOBSTER CSVs into `data/raw/` (git-ignored). First use builds an `.npy` cache.

```bash
python -m pytest tests -q
```
214 tests, about 140 seconds. Tests needing data skip themselves if it is absent.

```bash
python scripts/calibrate.py
```
Regenerates `configs/calibration.json` (per-stock impact coefficients).

**The trained models are committed** (`checkpoints/`, about 17 MB), so every evaluation runs without retraining:

```bash
python evaluation/basic_eval.py --episodes 20
python watchdog/dataset.py --episodes 150
python watchdog/evaluate_watchdog.py
python watchdog/lateburst_test.py
python watchdog/multiseed_summary.py --tags main main_s2 main_s3
```

Roughly 35 s, 2–4 min, 3 min, 1 min and 8 min respectively.

Retraining (only if you change the environment): each Spoofer is about 15 minutes at 600k steps, the Watchdog about 30 minutes at 400k steps. Five Spoofers train in parallel comfortably on 8 threads.

**Every result reproduces exactly from a fresh clone** (committed code and checkpoints, raw data copied in): 2,178 numeric values compared across all 12 scripts, 0 differences, plus all 32 Watchdog dataset files bit-for-bit and `configs/calibration.json` identical. Re-run any script and diff its `results/*.json` against `git show HEAD:<path>` (ignore `runtime_s`).

---

## 4. Repo map

```
env/            the replay simulation: data loading, feature normalization, price impact, LimitOrderBookEnv
training/       PPO training for the Spoofer roster + a training-trend summarizer
evaluation/     agents (trained + scripted), episode recorder, rule-based detector, basic evaluation
watchdog/       dataset recorder, Watchdog env/training, evaluation vs baseline, pre-registered tests,
                multi-seed CIs, PnL suppression, feature ablation
synthetic/      agent-based market: order book, background traders, calibration, spoof impact,
                env adapter, robustness evaluation, layering attacker + pre-registered test
scripts/        calibration, PnL decomposition, feature-signal diagnostic, real-case statistics
tests/          214 tests
results/        every reported number (.md to read, .json for exact values)
app.py          SPOOFWATCH CONSOLE: three views — overview, recorded demonstration, evidence
                (run: streamlit run app.py --server.fileWatcherType none — the flag is
                required because the repo's watchdog/ package shadows the PyPI one)
REPORT.md       full write-up: method, results, discussion, limitations, references
dashboard.html  self-contained results page built from results/ by scripts/build_dashboard.py
configs/        calibration files (committed)
checkpoints/    trained models (committed; other tags are local only)
```

`README.md` documents every script, its command, and how to read each report.

---

## 5. What every result file answers

| File | Question | Headline |
|---|---|---|
| `basic_eval.md` | Do the Spoofers make money, and how weak is the rule? | INTC +$13.2k, MSFT +$2.4k per episode; rule precision 0.20 on the RL pool |
| `watchdog_eval.md` | Watchdog vs rule on the same orders | F1 0.77 vs 0.35; FPR 0.07 vs 0.92 |
| `multiseed.md` | Do the headlines survive the training seed? | Yes: F1 0.753 ± 0.058; recall varies most |
| `lateburst_test.md` | Pre-registered: unseen attacker | Recall 0.24 vs 0.89; stock-shortcut hypothesis not supported |
| `layering_test.md` | Pre-registered: layered orders (synthetic market) | Watchdog F1 0.44 vs rule 0.73 |
| `pnl_suppression.md` | Does surveillance actually cut profit? | 83% vs 55%, with 12× fewer wrongful interventions |
| `feature_ablation.md` | What does the Watchdog use? | Trade direction: remove it and F1 collapses to 0.113 |
| `real_case_comparison.md` | How close is this to Coscia and Sarao? | Structure similar; speed and layering differ |
| `synthetic_eval.md` | Does any of it survive a different market mechanism? | Detection partly; profitability not at all |
| `synthetic_spoof_impact*.md` | How hard does a spoof move an emergent market? | 0.29–0.55 spreads vs the replay model's ~2.5 |
| `real_case_stats.md` | Our attackers' order structure | Cancel rates, resting times, side flips |
| `spoofer_seeds.md` | Does the Spoofer result depend on the training seed? | 3/3 seeds spoofed on both stocks; INTC stable, MSFT varies widely |
| `label_observability.md` | **How hard is the detection task really?** | **Not hard: a zero-parameter two-feature signature scores order F1 0.997 / 0.995 held-out. Read this before quoting any detector number.** |
| `supervised_baseline.md` | Is RL necessary for the detector? | No: gradient boosting beats it on every bucket (0.96 vs 0.77 in-distribution) |
| `temporal_split/comparison.md` | Did the train/test overlap inflate the results? | By 0.031 F1 in-distribution, below the ±0.058 seed spread; held-out buckets not inflated at all |

---

## 6. History worth knowing, so it is not repeated

Four environment exploits were found and closed. Each was caught by decomposing where PnL came from, never by looking at headline profit:

1. **Free accumulation** — the agent bought about 20× touch depth at a spoof-depressed price, filling below the real bid. Fixed by charging its own market orders price impact. (+$253k per episode before the fix.)
2. **Alternating-side swing** — linear permanent self-impact is path-independent, so swinging ±40 lots cost nothing while the spoof's shift applied to unlimited volume. Fixed by making spoof impact build up with order age and capping the favourable shift at the spoof's own size. (About $240k per episode before.)
3. **Early termination** — a run-over breached the inventory limit and ended the episode, letting the agent escape losses. All 31 of 31 early terminations happened on a run-over step. Fixed by scaling lot size to book depth and removing termination.
4. **Thin opening book** — episodes starting too early produced meaningless spoof sizes. Fixed with a 30,000-event warmup.

Two training failures and their causes:

- **Spoofers converged to never trading.** Not a missing opportunity: a scripted probe showed spoofing paid about $4k per episode. The cause was credit assignment — the roughly 50-event wait before trading was 50 decisions away. Fixed by one decision per 10 market events plus an entropy bonus.
- **The Watchdog collapsed to never flagging**, at entropy 0, 0.01, 0.03 and 0.1. The cause was input scale: book features sit near ±20 while the informative participant features are 0–5. Fixed by standardising observations (`--norm-obs`). **Its default flags still reproduce the collapse, so always pass `--norm-obs --ent-coef 0.01`.**

Two synthetic-market dead ends: a fixed cancel probability let liquidity grow without bound (the mid never moved), and thin-tailed market orders left the mid frozen in all 36 settings of the first calibration grid. Heavier-tailed order sizes fixed it.

---

## 7. Open issues, in the order I would tackle them

1. **The detector half of the project does not pose a hard problem** (§2b). CP-7 confirmed this rather than
   refuting it: it closed the gap to a ceiling that a zero-parameter rule already occupied. The fix is not a
   better learner — it is to withhold per-participant attribution from the observation, so the detector must
   infer manipulation from aggregate order flow. That changes the environment and requires retraining, and it is
   the only change that would make the detector results mean something. *Biggest item on this list.*
2. **Profitability does not transfer across market mechanisms.** In the synthetic market a spoof moves the price 4.5–9× less and every manipulator loses money. Either the replay impact model is too generous or the synthetic market is too inert (its only reactive channel is imbalance-following — no market makers repricing quotes). Deciding which would change how claim 1 is written. *Largest open question.*
3. **The Watchdog misses sparse spoofing** (a wait, then a few trades): recall 0.24–0.30 on both pre-registered attackers. Ablation says it keys on trade direction. Worth trying: train against a population that includes sparse styles, or add features summarising recent order lifecycle rather than instantaneous trading.
4. **Spoofing is unprofitable on wide-spread stocks** (AAPL, GOOG, AMZN), so those Spoofers never learned it and the held-out RL test (SPOOFER-05, AMZN) has no manipulation to catch. A held-out 1-tick stock would be better, but the dataset only has MSFT and INTC of that type and both are training stocks.
5. **Layering cannot be learned**, only scripted: the replay environment allows one large order per side. Supporting several would need an action-space change and full retraining.
6. **Seeds.** The two profitable Spoofers and the Watchdog now have 3 seeds each. MSFT varies widely across seeds (+$3,628 to +$15,886), so the committed `main` seed understates it; INTC is stable. SPOOFER-01/03/05, which never learned to spoof, remain single-seed.
7. **The manipulation label is a proxy** — trading on the opposite side while the order rests. It stands in for intent; it is not a legal determination.

---

## 8. Reproduction status

Complete. Every script was re-run in a clean clone and every output matched the committed results exactly
(2,178 numeric values, 0 differences; the Watchdog dataset matched bit-for-bit). Nothing is outstanding here.

One protocol defect was found by audit afterwards and fixed: episode starts were drawn from the whole trading
day for every split, so 33 of 50 test episodes overlapped a training episode. Splits now get disjoint windows
(`watchdog/dataset.py --time-split`). Its measured cost was **0.031 order F1 in-distribution** — below the
±0.058 seed spread — and **nothing on the held-out buckets**, so the published numbers stand and remain the
default. `results/temporal_split/comparison.md` has the three-condition table.

## 9. Working rules that kept this honest

Worth keeping:

- **Pre-register held-out tests.** Attacker, procedure and hypothesis are committed *before* any data exists, and the test runs once. Both such tests went against the Watchdog and were reported that way. Once a held-out attacker has been inspected during analysis it is no longer a clean test — that is why LATEBURST and LAYER-ATK exist.
- **Never tune on held-out data.** One design decision (dropping book features) was evaluated on the test split only and rejected on that evidence.
- **Decompose profit before believing it.** `scripts/pnl_decompose.py` separates spoof gain from spread cost, self-impact and drift, with an impact-free counterfactual. It caught two exploits that headline PnL hid.
- **Every documented number comes from a committed script**, not from memory or estimation.
- **Record negative results.** The failed calibration rules, collapsed training runs and rejected changes are all in the README and in code comments.

---

## 10. Gotchas

- **Data and Watchdog datasets are git-ignored**; models are committed. A fresh clone needs `data/download.py` (~600 MB) and `watchdog/dataset.py` before the Watchdog evaluations.
- **`multiseed_summary.py` needs the LATEBURST data**, so run `lateburst_test.py` first. It is seeded, so rerunning regenerates identical data and does not revisit the pre-registered verdict.
- **Run trained Spoofers with sampled actions** (`ModelPolicy(path, deterministic=False)`). With the most likely action, SPOOFER-02 loses its learned spoofing entirely.
- **Units**: agents decide once per 10 market events. Order lifetimes are recorded in decisions, but the rule compares them in market events (`OrderRecord.lifetime_events`).
- **LOBSTER has no participant identities**, so the Watchdog's features cannot be built from real data. Only the rule can run on real order flow. SPY was never used (it also covers only 09:30–10:30).
- **Windows PowerShell and git**: double quotes inside commit messages break argument parsing. Use here-strings and avoid `"` in messages.
- The environment is a single trading day. There is no second day in the LOBSTER sample release.

---

## 11. If you only read one thing

Read `docs/FINAL_AUDIT.md` §C, then §2b and §2c above. The detector half of this project does not pose a hard problem: the label
is readable from two of the detector's own input features, a zero-parameter signature scores 0.997, and
gradient boosting beats the RL detector everywhere. Present the detector work as a documented negative result.

What stands up, and is worth presenting: **RL agents discovering profitable spoofing on their own**, under a
price-impact model calibrated from real order flow, with structure matching documented enforcement cases — plus
a simulator, a calibration and a reproduction pipeline that are correct and verified. After that, run
`streamlit run app.py --server.fileWatcherType none` (or open `dashboard.html`) and read `REPORT.md`.
