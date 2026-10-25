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
| 2. The Watchdog generalises to unseen attacks | **Not shown** | Two pre-registered tests, each run once: LATEBURST recall 0.24 vs rule 0.89; layering recall 0.30 vs rule 1.00. |
| 3. It does not flag legitimate activity | **Shown in simulation** | False-positive rate on legitimate orders 0.031 ± 0.067 across 3 seeds vs the rule's 0.823; 0.11 vs 0.92 in the synthetic market. Not testable on real data (see §10). |

Supporting results:

- **In-distribution detection**: Watchdog order-level F1 **0.753 ± 0.058** (3 seeds) vs rule **0.352**. It flags while the order still rests (median delay 0 steps); the rule can only fire at the cancel (8 steps).
- **PnL suppression**: with the Watchdog acting inside episodes (flag → cancel the trader's resting orders, block new ones for 200 events), SPOOFER-04's profit falls **83%** vs the rule's 55%, while intervening on legitimate traders about 12× less often.
- **Interpretability**: detection rests almost entirely on one feature. Ablating "trade direction" drops F1 from 0.772 to **0.113**. That is also why sparse spoofing is missed.
- **Real cases**: structurally similar to Coscia in the dimensions the agents choose (large orders one side, trading the other, fills 0.4–1.4% vs his 0.08–0.5%, side alternation). Different in speed (44–52% of our spoofs rest over 1 s vs his 0.57%) and no layering in the replay environment.

---

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
115 tests, about 30 seconds. Tests needing data skip themselves if it is absent.

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
tests/          115 tests
results/        every reported number (.md to read, .json for exact values)
app.py          Streamlit dashboard: inputs, backend pipeline, results, evidence
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

1. **Profitability does not transfer across market mechanisms.** In the synthetic market a spoof moves the price 4.5–9× less and every manipulator loses money. Either the replay impact model is too generous or the synthetic market is too inert (its only reactive channel is imbalance-following — no market makers repricing quotes). Deciding which would change how claim 1 is written. *Largest open question.*
2. **The Watchdog misses sparse spoofing** (a wait, then a few trades): recall 0.24–0.30 on both pre-registered attackers. Ablation says it keys on trade direction. Worth trying: train against a population that includes sparse styles, or add features summarising recent order lifecycle rather than instantaneous trading.
3. **Spoofing is unprofitable on wide-spread stocks** (AAPL, GOOG, AMZN), so those Spoofers never learned it and the held-out RL test (SPOOFER-05, AMZN) has no manipulation to catch. A held-out 1-tick stock would be better, but the dataset only has MSFT and INTC of that type and both are training stocks.
4. **Layering cannot be learned**, only scripted: the replay environment allows one large order per side. Supporting several would need an action-space change and full retraining.
5. **Seeds.** The two profitable Spoofers and the Watchdog now have 3 seeds each. MSFT varies widely across seeds (+$3,628 to +$15,886), so the committed `main` seed understates it; INTC is stable. SPOOFER-01/03/05, which never learned to spoof, remain single-seed.
6. **The manipulation label is a proxy** — trading on the opposite side while the order rests. It stands in for intent; it is not a legal determination.

---

## 8. Reproduction status

Complete. Every script was re-run in a clean clone and every output matched the committed results exactly
(2,178 numeric values, 0 differences; the Watchdog dataset matched bit-for-bit). Nothing is outstanding here.

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

Open `dashboard.html` in a browser, then read "Results at a glance" at the top of `README.md`, then `results/watchdog_eval.md` and `results/synthetic_eval.md`. Between them they hold the honest answer: the Watchdog is far better than a static rule at not crying wolf, and no better — usually worse — at catching manipulation it has not seen.
