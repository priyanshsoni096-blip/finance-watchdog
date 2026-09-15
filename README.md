# RL Finance Watchdog

**An Adversarial Reinforcement Learning System for the Detection of Limit Order Book Spoofing**

> Can a reinforcement learning agent (the "Watchdog") learn to detect *previously unseen* order-book manipulation strategies — specifically spoofing — while maintaining a low false-positive rate on legitimate market activity, without being given explicit rules about what spoofing looks like?

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

No labeled real-world spoofing dataset exists to validate detection recall against, so this project makes no claim of real-world detection capability.

Legal reference: CEA §4c(a)(5)(C) (Dodd-Frank §747).

---

## Status

| Piece | State |
|---|---|
| Data loader, cross-stock normalization, price-impact calibration | Done, tested on real data |
| `LimitOrderBookEnv` (spoof impact, self-impact, run-over, liquidation) | Done, tested — **one open modelling issue** (see Open issues) |
| Spoofer population SPOOFER-01..05 (PPO) | Trained on the current env; see `results/basic_eval.md` |
| Rule-based baseline detector | Done |
| Basic evaluation (`evaluation/basic_eval.py`) | Done |
| Watchdog: dataset recorder, env, RecurrentPPO training, evaluation vs baseline | Done: normalised observations, entropy 0.01; `results/watchdog_eval.md` |
| Pre-registered clean held-out test (LATEBURST-ATK) | Done: `results/lateburst_test.md` |
| Multi-seed Watchdog CIs (3 seeds) | Done: `results/multiseed.md` (each Spoofer is still one seed) |
| PnL suppression with surveillance acting inside episodes | Done: `results/pnl_suppression.md` |
| Interpretability (feature-group ablation) | Done: `results/feature_ablation.md` |
| Structural comparison with Coscia and Sarao | Done: `results/real_case_comparison.md` (speed and layering cannot match in this environment) |
| Coscia-style layering attacker (needs several resting orders per side), synthetic agent-based market, Watchdog on real SPY flow | Not started |

---

## How to run

All commands run from the repo root. Tested on Windows 11, Python 3.13, CPU only.

```bash
pip install -r requirements.txt
```

```bash
python data/download.py
```
Downloads ~600 MB of LOBSTER sample CSVs (AAPL, MSFT, GOOG, INTC, AMZN, 10 levels) into `data/raw/`. The first load builds a `.npy` cache in `data/cache/`.

```bash
python scripts/calibrate.py
```
Measures per-stock spread, depth and the price-impact coefficient; writes `configs/calibration.json`.

```bash
python -m pytest tests -q
```
88 tests, about 1–2 minutes. Tests that need data skip themselves if it isn't downloaded.

```bash
python training/train_spoofer.py SPOOFER-04 --timesteps 600000
```
Trains one Spoofer (roster below). Output goes to `checkpoints/SPOOFER-04/main/`: `model.zip`, `progress.csv` (one row per 4,000 steps), `config.txt`. About 25 minutes at 600k steps on this laptop. `--threads`, `--seed`, `--tag` are optional.

```bash
python training/summarize.py SPOOFER-04 --buckets 10
```
Prints how behaviour changed over training: reward, PnL, action mix, trades against the agent's own resting spoof, run-overs.

```bash
python evaluation/basic_eval.py --episodes 20
```
Runs the basic evaluation (about 10–15 minutes) and writes:
- `results/basic_eval.md` — readable report with the tables described below
- `results/basic_eval.json` — every number, including the full detector tuning grid

Use `--episodes 5` for a quick look. Spoofers without a checkpoint are skipped with a message.

### Watchdog pipeline

Run these after the Spoofers are trained, in order.

```bash
python watchdog/dataset.py --episodes 150
```
Rolls out every frozen agent once and saves what surveillance can see into `data/watchdog/{train,test,heldout}/`: 40 book features plus 6 features of the watched trader's own orders, with hindsight labels per step. SPOOFER-05, SCRIPTED-ATK and every AMZN source go only into `heldout`.

```bash
python watchdog/train_watchdog.py --timesteps 400000
```
Trains the Watchdog (RecurrentPPO, LSTM) on the `train` split. At each step it chooses to flag or clear: +1 for a correct flag, −1 for a miss, −2 for a false flag, 0 for correctly clearing. It is never told the spoofing rule. Output goes to `checkpoints/WATCHDOG/main/` (`model.zip`, `progress.csv` with precision, recall and false-positive rate per rollout).

```bash
python watchdog/evaluate_watchdog.py
```
Scores the Watchdog and the rule-based baseline on the same recorded orders; writes `results/watchdog_eval.md` and `.json`. Order-level precision, recall, F1 and false-positive rate per bucket, plus how many steps each takes to flag. The rule can only fire on the cancel; the Watchdog can flag while the order still rests.

```bash
python watchdog/lateburst_test.py
```
The pre-registered clean held-out test: a new attacker defined before any of its data existed. **Run it once**; its procedure and hypothesis are fixed. Output: `results/lateburst_test.md` and `.json`.

```bash
python watchdog/multiseed_summary.py --tags main main_s2 main_s3
```
Scores every trained Watchdog seed on the same test, heldout and saved LATEBURST-ATK data. Reports each headline metric per seed, with its mean and a 95% Student-t CI, plus the seed-independent rule. Output: `results/multiseed.md` and `.json`. Train extra seeds with `python watchdog/train_watchdog.py --timesteps 400000 --norm-obs --ent-coef 0.01 --tag main_s2 --seed 11`.

### Reading `results/basic_eval.md`

- **C1 table** — one row per agent run. Compare a trained Spoofer's PnL with HONEST and SCRIPTED-ATK on the same stock. "Trades against own spoof/ep" is the spoofing signature (spoof one side, trade the other). "PnL (spread·lots)" makes stocks comparable. Look at the CI: if it crosses zero, the profit isn't established.
- **Detector table** — the rule-based baseline, tuned on training stocks only, scored per bucket. "Manipulative" orders are ones whose owner traded on the opposite side while they rested. The held-out AMZN rows are the generalisation check. The "Flicker only" row is the false-positive check on large-order-then-cancel with no trading.
- **Real-flow table** — the same rule on real LOBSTER messages. There are no labels, so this is a flag *rate*, not a false-positive rate.

---

## What is what

```
data/download.py                  fetch raw LOBSTER CSVs from Hugging Face
env/lobster_data.py               parse message + orderbook CSVs into arrays (LobsterDay), cache
env/normalization.py              scale-free features: prices in spread units, sizes as log ratio to trailing size
env/price_impact.py               linear/sqrt impact; lambda and impact ramp fitted from order-flow imbalance
env/lob_env.py                    LimitOrderBookEnv (Gymnasium): actions, fills, spoof ramp + cap, self-impact,
                                  run-over, liquidation, reward
configs/calibration.json          per-stock lambda and impact ramp written by scripts/calibrate.py

training/train_spoofer.py         PPO training for the SPOOFER roster (DECISION_ENV: 10 events per decision)
training/summarize.py             training trend table from progress.csv

evaluation/agents.py              policies: trained Spoofer (ModelPolicy), HONEST, FLICKER, SCRIPTED-ATK
evaluation/rollout.py             one recorded episode: PnL, order lifecycles, 46-dim surveillance view, labels
evaluation/baseline_detector.py   rule-based detector, metrics, tuning, real-data front-end
evaluation/basic_eval.py          Spoofers vs controls, rule-based baseline, real-flow flag rate

watchdog/dataset.py               record frozen agents into train / test / heldout splits (data/watchdog/)
watchdog/watchdog_env.py          WatchdogEnv (flag/clear rewards), flag_episode, observation normalisation
watchdog/train_watchdog.py        RecurrentPPO Watchdog training on the train split
watchdog/evaluate_watchdog.py     Watchdog vs rule-based baseline on identical orders
watchdog/lateburst_test.py        pre-registered clean held-out test (LATEBURST-ATK), run once
watchdog/multiseed_summary.py     headline metrics across Watchdog training seeds with 95% CIs
watchdog/pnl_suppression.py       Watchdog and rule acting inside episodes: PnL suppression, wrongful interventions
watchdog/feature_ablation.py      which observation features the Watchdog relies on (group ablation, test split)

synthetic/market.py               price-time-priority limit order book (synthetic market, evaluation only)
synthetic/simulator.py            background traders: noise traders and imbalance followers
synthetic/calibrate.py            grid search against real MSFT/INTC market statistics -> configs/synthetic_calibration.json
synthetic/spoof_impact.py         emergent price impact of a spoof vs an identical control market

scripts/calibrate.py              per-stock statistics -> configs/calibration.json
scripts/pnl_decompose.py          where a Spoofer's PnL comes from (spoof gain vs costs, no-impact counterfactual)
scripts/feature_signal_check.py   diagnostic: can a simple supervised model separate positives from the features
scripts/real_case_stats.py        measured attacker order structure for the real-case comparison

tests/                            unit and real-data tests (data, normalization, impact, env, detector, rollout,
                                  Watchdog env/eval, roster granularity)
checkpoints/                      trained models + logs (git-ignored)
results/                          basic_eval, watchdog_eval, lateburst_test, multiseed, pnl_suppression,
                                  feature_ablation, real_case_stats, synthetic_spoof_impact (.md and .json each);
                                  real_case_comparison.md
```

### Agent roster

| Agent | Stock | Role | Variation |
|---|---|---|---|
| SPOOFER-01 | AAPL | training pool | high inventory penalty (0.01) |
| SPOOFER-02 | MSFT | training pool | low inventory penalty (0.0001) |
| SPOOFER-03 | GOOG | training pool | can only spoof the bid |
| SPOOFER-04 | INTC | training pool | default |
| SPOOFER-05 | AMZN | **held out** | default; never used for tuning |
| HONEST, FLICKER, SCRIPTED-ATK | all | scripted controls in the eval | legitimate trading / large orders cancelled with no trading / spoof one side, trade the other |

### Environment in one paragraph

Each step replays one real LOBSTER event. The agent can do nothing, market buy or sell one lot, place a large "spoof" order at the best bid or ask (10 × trailing touch depth), or cancel. A resting spoof shifts the price the agent trades at by `lambda × spread × size / depth`. Its own market orders push the price too (self-impact). A spoof's effect builds up over its first ~200 events, and only as many shares as the spoof's size can trade at the favourable price. The position is valued at the *real* historical mid and liquidated across the spread at the end. A spoof is filled ("run over") if the real market moves through its price. Reward is the PnL change in units of spread × lot, minus an inventory penalty. Lot = half the stock's median touch depth; episodes are 2,000 events and start after event 30,000.

---

## Measured calibration (`python scripts/calibrate.py`)

| Ticker | Events | Median mid | Median spread | Spread (ticks) | Touch depth | OFI λ | OFI R² | 10×-depth spoof moves mid by |
|---|---|---|---|---|---|---|---|---|
| AAPL | 400,391 | $583.24 | $0.15 | 15 | 150 | 0.046 | 0.40 | 0.46 spreads |
| MSFT | 668,765 | $30.51 | $0.01 | 1 | 13,318 | 0.255 | 0.73 | 2.55 spreads |
| GOOG | 147,916 | $569.88 | $0.27 | 28 | 138 | 0.062 | 0.46 | 0.62 spreads |
| INTC | 624,040 | $27.02 | $0.01 | 1 | 13,915 | 0.247 | 0.66 | 2.47 spreads |
| AMZN | 269,748 | $222.66 | $0.13 | 13 | 164 | 0.037 | 0.33 | 0.37 spreads |

---

## Design changes from the handoff, and why

Each was driven by a measurement.

1. **Stable-Baselines3 instead of Ray RLlib.** CPU-only Windows laptop, Python 3.13; Ray on Windows is beta. Network and hyperparameters unchanged.
2. **Prices in spread units, not ticks.** Median spread is 15 ticks on AAPL and 1 tick on INTC; in spread units both give a median best-level offset of 0.500.
3. **Spoof size = 10 × trailing touch depth**, not a fixed 10,000 shares (touch depth ranges ~140 to ~13,900 shares).
4. **Impact λ fitted per stock from order-flow imbalance** (Cont, Kukanov & Stoikov 2014). The handoff's percentile rule was implemented and rejected: it implied a 10×-depth spoof moves MSFT/INTC by 44–58 spreads.
5. **Position valued at the real (unimpacted) mid**, so holding while a spoof rests earns nothing. Test: the reward stream is identical with or without the spoof.
6. **Reward in spread × lot units** so stocks are comparable.
7. **Lot scaled to depth; no early termination.** With 100-share lots, spoofs were 1,333–1,396 lots on MSFT/INTC and all 31/31 early terminations happened on a run-over step, an escape from losses. Now 0 terminations.
8. **End-of-episode liquidation across the spread**, including unwinding impact.
9. **30,000-event warmup.** INTC's trailing depth is 0.08× its day median at event 15,000; a 5,000 warmup gave spoofs under 5 lots on 3.9% of INTC starts, 30,000 gave none.
10. **Self-impact for the agent's own market orders.** Without it, the first SPOOFER-04 bought 276,900 shares (about 20× touch depth) at a spoof-depressed price, filling below the real bid, and made about +$253k per episode. Test: once bought volume reaches the spoof size, further buys fill at or above the ask.
11. **Spoof impact builds up with order age, and its benefit is capped at the spoof's size.** After change 10, SPOOFER-02/04 still made about $240k per episode by swinging between −40 and +40 lots under alternating spoofs. Linear self-impact is path-independent, so a round trip cost only $53, while the spoof's shift applied to unlimited volume.
    - The build-up follows the measured price response to order flow at 10/50/200/1000 events. MSFT is at 0.57 of its 200-event response after 10 events, INTC 0.55, AMZN 0.49; AAPL and GOOG show no build-up. The age-0 value of 0 is an assumption, since the smallest measured window is 10 events.
    - Each spoof's favourable shift applies to at most its own size in opposite-side shares; adverse shifts always apply.
    - A per-trade cost was considered and dropped: lots fill within the best level, where the measured walking cost is zero.
    - Replaying the v2 policies for 5 episodes: MSFT **+$232,119 → −$8,052**, INTC **+$255,487 → −$6,905** (fix off vs on).

---

## Findings so far

- **First training round (before fixes 7 and 10):** SPOOFER-01 (AAPL) converged to never trading — PnL and reward exactly 0, spoofs placed and cancelled after a median of 3 steps. On AAPL the calibrated spoof shift ($0.068) is smaller than half the spread ($0.075), so spoofing can't pay. SPOOFER-04 (INTC) found the free-accumulation exploit described in change 10.
- **Third training round (after change 11, 600k steps): every Spoofer converged to never trading.** Market buys and sells were ≤0.2% of actions, and final PnL ranged from −$0.38 to −$302 per episode.
  - **Profitable spoofing still exists in this env.** A scripted probe (spoof, wait W events, trade k lots on the other side, cancel, unwind; 6 episodes per setting) found:

    | Stock | Wait 0 | Wait 50 | Wait 200 |
    |---|---|---|---|
    | INTC | loses on every run | **+$3,980** (5 lots, 83% profitable) | break-even |
    | MSFT | loses on every run | **+$4,043** (3 lots, 100% profitable) | break-even |
    | AAPL | loses | loses (every setting) | loses |

  - So the Spoofers' failure is an exploration and credit-assignment problem, not a missing opportunity. Random trading early in training cost about $13.6k per episode on INTC, and the default entropy bonus is 0, so the policy settled on not trading before it found the delayed payoff.
- **Exploration experiments (300k steps, one seed each)** — fix for the no-trade convergence:

  | Setting | Stock | Deterministic PnL/episode | Spoof gain | No-impact counterfactual |
  |---|---|---|---|---|
  | entropy 0.01, 1 event/step | INTC | −$1,582, 0 trades | $0 | $0 |
  | 10 events/step | INTC | **+$6,477**, 187 trades | +$16,920 | −$10,267 |
  | 10 events/step + entropy 0.01 | INTC | **+$8,236**, 186 trades | +$17,365 | −$8,761 |
  | 10 events/step + entropy 0.01 | MSFT | $0, 0 trades | $0 | $0 |

  - Deciding once per 10 market events makes the ~50-event wait before trading a 5-decision delay instead of 50, and INTC learns genuine spoofing. The same trades with impact switched off lose money, so the profit comes from the spoof, not an exploit.
  - MSFT still fails: +$881/episode in training with sampled actions, but the deterministic policy never trades.
  - All agents now use 10 events per decision and entropy 0.01 (`DECISION_ENV` in `training/train_spoofer.py`).
- **Final Spoofer population (10 events/step, entropy 0.01, 600k steps).** PnL breakdown over 10 episodes each, sampling actions from the trained policy:

  | Spoofer | Stock | PnL/episode | Spoof gain | No-impact counterfactual | Most-likely-action PnL |
  |---|---|---|---|---|---|
  | SPOOFER-04 | INTC | **+$13,655** (171 trades) | +$22,086 | −$8,414 | +$10,618 |
  | SPOOFER-02 | MSFT | **+$2,085** (108 trades) | +$7,664 | −$5,407 | −$1,291 (no spoof gain) |
  | SPOOFER-03 | GOOG | +$126 (12 trades) | +$182 | −$18 | +$202 |
  | SPOOFER-01 | AAPL | −$91 (essentially no trades) | ~$0 | — | +$199 (0 trades) |
  | SPOOFER-05 | AMZN | $0 (no trades) | $0 | — | $0 |

  - INTC and MSFT learned genuine spoofing: with impact switched off, the same trading loses money.
  - Taking the most likely action erased MSFT's learned behaviour, so evaluation and the Watchdog dataset sample actions from the policy PPO actually optimised.
  - **The held-out SPOOFER-05 (AMZN) never spoofs**, because spoofing isn't profitable on AMZN under the calibrated impact. The held-out RL test therefore has no manipulation to catch; the scripted attacker on AMZN is the held-out check.
- **Watchdog dataset (`python watchdog/dataset.py --episodes 150`, 213 s, 52 MB).** Share of steps with a manipulative order resting:

  | Split | Source | Episodes | Positive steps |
  |---|---|---|---|
  | train | SPOOFER-04 INTC | 150 | 92.4% |
  | train | SPOOFER-02 MSFT | 150 | 44.1% |
  | train | SPOOFER-03 GOOG | 150 | 20.2% |
  | train | SPOOFER-01 AAPL | 150 | 0.2% (7,877 large orders, almost none traded against) |
  | train | HONEST / FLICKER on AAPL, MSFT, GOOG, INTC | 150 each | 0% |
  | test | same sources, unseen seeds | 50 each | within about 1 point of train |
  | heldout | SCRIPTED-ATK on all five stocks | 50 each | 51–57% |
  | heldout | SPOOFER-05 AMZN, HONEST / FLICKER AMZN | 50 each | 0% |

  The held-out RL source has no positives, so on AMZN only false positives can be measured.
- **The surveillance features carry the signal (`python scripts/feature_signal_check.py`).** This check is supervised, so it is a diagnostic, not a detector result. A class-balanced logistic regression on [current step, mean of the previous 10 steps], trained on the `train` split, step level:

  | Scored on | Precision | Recall | False-positive rate |
  |---|---|---|---|
  | test: SPOOFER-02/03/04 | 0.94 | 0.95 | 0.07 |
  | test: legitimate only (HONEST + FLICKER) | — | — | 0.19 |
  | test: SPOOFER-01 (large orders, no manipulation) | 0.01 | 0.29 | 0.14 |
  | heldout: SCRIPTED-ATK, all stocks | 0.98 | 0.49 | 0.009 |
  | heldout: AMZN legitimate sources | — | — | 0.09 |

  If the Watchdog fails to separate positives, the cause is RL optimisation, not missing information. Even a simple model loses half its recall on the held-out scripted attacker's unseen structure.
- **First Watchdog training run collapsed to never flagging.** Settings: RecurrentPPO, entropy 0, 400k steps planned. Flag rate over rollouts:

  | Steps | 2k | 10k | 16k | 32k and later |
  |---|---|---|---|---|
  | Flag rate | 0.49 | 0.07 | 0.02 | **0.000** |

  - Early random flags land mostly on legitimate steps at −2 each. Never flagging costs only the missed positives (about −0.13 per step), so it wins before the features are learned.
  - The run was stopped at 49k steps.
- **An entropy bonus does not fix it.** Three runs at entropy 0.01, 0.03 and 0.1 (150k steps each, judged on training-rollout metrics only) all collapsed the same way:

  | Steps | 20k | 40k | 60k and later |
  |---|---|---|---|
  | Flag rate | 0.005–0.028 | 0.001 | 0.000 (recall 0.000 at the end) |

  - Suspected cause: input scale. Book price features sit near ±20, while the participant features that carry the signal are 0–5. The logistic-regression diagnostic standardised its inputs; the Watchdog did not.
- **Observation normalisation fixes it; the asymmetric false-positive cost was not the cause.** Three runs, 150k steps each, entropy 0.01. Metrics are over the last five training rollouts that contained positives:

  | Run | Inputs | FP cost | Precision | Recall | FPR |
  |---|---|---|---|---|---|
  | norm_fp2 | normalised | −2 | 0.92–1.00 | 0.27–0.73 | ≤0.003 |
  | norm_fp1 | normalised | −1 | 0.70–1.00 | 0.20–0.72 | ≤0.010 |
  | raw_fp1 | raw | −1 | collapsed | 0.000 | ~0 |

  - The raw-input run still collapsed at the milder −1 cost, so the input scale is what blocked learning.
  - The Watchdog is trained with normalised observations, FP cost −2 (the design) and entropy 0.01. The setting was chosen on training-rollout metrics only; the test and heldout splits stay unseen.
- **Final Watchdog (400k steps) vs rule-based baseline, scored on identical orders (`results/watchdog_eval.md`).**
  - The rule is tuned on the train split: size ≥ 2× depth, cancelled within 1,000 events.
  - Training ended with precision 0.94–0.99 (one rollout at 0.66), recall 0.68–1.00 and FPR ≤ 0.022 over the last 8 rollouts.

  Order level:

  | Bucket | Watchdog P / R / F1 / FPR | Rule P / R / F1 / FPR | Median delay (steps), Watchdog / rule |
  |---|---|---|---|
  | In-distribution (test split) | **0.75 / 0.80 / 0.77 / 0.07** | 0.22 / 0.94 / 0.35 / 0.92 | **0** / 8 |
  | Held-out SPOOFER-05 + AMZN (all legitimate) | FPR **0.01** | FPR 0.97 | — |
  | Legitimate only (HONEST + FLICKER) | FPR **0.04** | FPR 0.82 | — |
  | Held-out scripted attacker, train stocks | 0.83 / **0.25** / 0.38 / 0.04 | 0.51 / **0.94** / 0.66 / 0.77 | 6 / 11 |
  | Held-out scripted attacker, AMZN | 0.73 / **0.15** / 0.26 / 0.05 | 0.50 / **0.95** / 0.65 / 0.81 | 8 / 11 |

  - Step level on the test split: precision 0.95, recall 0.86, FPR 0.006.
  - Legitimate episodes with any flag: FLICKER 22–40%, HONEST 0–4%.
- **What the Watchdog misses is sparse spoofing, whichever source produces it.** Medians over manipulative orders, from the recorded datasets:

  | Source | Opposite trades per order | Trades per resting step | Watchdog |
  |---|---|---|---|
  | SPOOFER-04 INTC (train / test) | 12 / 11 | 0.92 | caught (flag rate 0.93 vs 0.93 positive) |
  | SPOOFER-02 MSFT | 4 / 5 | 0.67 | caught (0.45 vs 0.45) |
  | SPOOFER-03 GOOG | 1 / 2 | 0.33 / 0.40 | missed (0.018 vs 0.21) |
  | SCRIPTED-ATK, all stocks (heldout) | 3 | 0.25–0.27 | missed (0.011–0.075 vs 0.51–0.57) |

  Sparse spoofing (a wait, then a few trades) is already present in training through SPOOFER-03. The held-out miss is therefore not purely an unseen structure: the Watchdog under-learned a style it was shown.
- **Per-source feature diagnostic (`python scripts/feature_signal_check.py`, all 46 features).**

  | Scored on | Step recall | Step precision |
  |---|---|---|
  | test: SPOOFER-02 MSFT | 1.00 | 0.96 |
  | test: SPOOFER-03 GOOG | 0.67 | 0.64 |
  | test: SPOOFER-04 INTC | 1.00 | 1.00 |
  | heldout: SCRIPTED-ATK on MSFT / INTC | 0.90 / 0.91 | 0.99 / 0.99 |
  | heldout: SCRIPTED-ATK on AAPL / GOOG / AMZN | 0.17 / 0.17 / 0.22 | 0.97–0.98 |

  - At a 0.5 threshold, sparse GOOG spoofing reaches recall 0.67 supervised. Pooled with legitimate test activity, however, even the supervised model ranks it poorly (AUC 0.851, average precision 0.061 at a 2.1% positive rate; see below). So the Watchdog's miss there (flag rate 0.018 vs 0.21 positive) is not purely under-learning: sparse spoofing is only weakly separable from legitimate trading with these features.
  - The scripted attacker behaves identically on every stock, yet it is caught on MSFT and INTC and mostly missed elsewhere. The Watchdog's flag rates lean the same way (0.070–0.075 on MSFT/INTC vs 0.011–0.022 elsewhere).
  - Nearly all training positives come from the two 1-tick stocks, so book features likely act as a stock-identity shortcut.
  - **Method note:** this was found using the heldout split. Any Watchdog change it motivates will be chosen on test-split evidence only (`--no-heldout`), and its heldout numbers will be labelled post-hoc rather than a clean generalisation test.
- **Removing book features does not help (test split only, `--no-heldout`, threshold-free).** Each Spoofer's steps are pooled with all legitimate test steps:

  | Test bucket | All 46 features: AUC / AP | Participant 6 only: AUC / AP |
  |---|---|---|
  | SPOOFER-02 MSFT + legitimate | **0.955 / 0.378** | 0.876 / 0.147 |
  | SPOOFER-03 GOOG + legitimate | 0.851 / 0.061 | 0.858 / 0.068 |
  | SPOOFER-04 INTC + legitimate | **0.939 / 0.549** | 0.869 / 0.340 |
  | all test sources | **0.938 / 0.623** | 0.879 / 0.410 |

  - Participant-only raised GOOG recall at a 0.5 threshold (0.67 → 0.96) only by flagging more: legitimate FPR rose 0.19 → 0.34, and ranking was no better on GOOG and worse elsewhere.
  - **Decision: the Watchdog keeps all 46 features; no change is adopted.** The stock-identity shortcut remains an unverified hypothesis supported only by the heldout split. Testing it cleanly needs a fresh attacker variant that has not been inspected.

## Pre-registered clean held-out test: LATEBURST-ATK

The scripted attacker has been inspected during analysis, so it no longer gives a clean generalisation result. This test was defined and committed **before** any of its data was generated or any result was seen.

- **Attacker** (`LateBurstSpoof` in `evaluation/agents.py`). Spoof one side, wait 100–200 market events, trade 4–8 lots on the other side in a burst, cancel immediately, unwind. Identical behaviour on AAPL, MSFT, GOOG, INTC and AMZN.
- **Evaluation.** 50 episodes per stock at 10 events per step. The existing Watchdog (`checkpoints/WATCHDOG/main`) and the train-tuned rule are scored on the same orders. Negatives are FLICKER on the same stock (test-split seeds). No model, threshold or feature is changed after results are seen.
- **Hypothesis (stock-identity shortcut).** The Watchdog's order-level recall on MSFT+INTC is at least **2×** its recall on AAPL+GOOG+AMZN.
  - Supported if the ratio is ≥ 2.
  - Not supported if it is below 2.
  - Undetermined if recall is 0 on both groups.
- **Also reported, not used for any decision.** Overall Watchdog recall, precision and FPR versus the rule.

**Result (run once after the pre-registration commit; `results/lateburst_test.md`): hypothesis not supported.**

| Group | Watchdog P / R / FPR | Rule P / R / FPR | Median delay (steps) W / R |
|---|---|---|---|
| All stocks | **0.66 / 0.24 / 0.05** | 0.31 / **0.89** / 0.76 | 12 / 22 |
| MSFT + INTC | 0.75 / 0.30 / 0.04 | 0.33 / 0.90 / 0.81 | 0 / 22 |
| AAPL + GOOG + AMZN | 0.58 / 0.20 / 0.05 | 0.30 / 0.88 / 0.73 | 16 / 22 |

- **Shortcut hypothesis.** The recall ratio is 0.296 / 0.198 = **1.50**, below the committed threshold of 2. Recall is somewhat higher on the 1-tick stocks, but the stock-identity shortcut is not supported at the pre-registered bar.
- **Generalisation.** On a clean, unseen attacker the Watchdog catches 24% of manipulative orders versus the rule's 89%. It flags 5% of legitimate large orders versus the rule's 76%.
- **Attacker PnL per episode (context only).** MSFT +$442, INTC +$726; AAPL −$471, GOOG −$735, AMZN −$330.

## PnL suppression: surveillance acting inside the episode

`python watchdog/pnl_suppression.py --episodes 50` → `results/pnl_suppression.md`. Every source runs the same seeds under three conditions: no surveillance, the Watchdog, and the rule. On a flag, the exchange cancels the participant's resting large orders and rejects new ones for 200 market events. The rule can only fire when an order is cancelled.

| Source | PnL, no surveillance | PnL with Watchdog (suppression) | PnL with rule (suppression) | Trades against own spoof/ep: none / W / R |
|---|---|---|---|---|
| SPOOFER-04 INTC | +$12,625 ± 1,162 | **+$2,165 ± 485 (83%)** | +$5,655 ± 657 (55%) | 173 / **36** / 76 |
| SPOOFER-02 MSFT | +$3,186 ± 517 | **−$4,561 ± 1,275 (243%)** | +$140 ± 831 (96%) | 60 / **6** / 45 |

Interventions on legitimate activity, per episode:

| Source | Watchdog: interventions / blocked orders | Rule: interventions / blocked orders |
|---|---|---|
| FLICKER (MSFT, INTC) | **0.3–0.4 / 1.1–1.6** | 4.7 / 17.6–18.8 |
| HONEST (MSFT, INTC) | 0 / 0 | 0 / 0 |

- **Where profit without surveillance is clearly positive (the RL Spoofers), the Watchdog suppresses more than the rule**, and it intervenes on legitimate large orders about 12× less often.
- **The scripted and LATEBURST attackers were not reliably profitable without surveillance**: +$418 ± 956 and +$748 ± 634 (scripted, MSFT / INTC), +$531 ± 674 and −$87 ± 1,007 (LATEBURST). Their suppression percentages are therefore not meaningful. On them the rule drives PnL lower than the Watchdog does, consistent with its higher recall on unseen structure.
- **Suppression above 100%** means the intervention turned profit into a loss.
- **The attackers are frozen and do not adapt to surveillance.** These numbers measure suppression of non-adaptive manipulation only.

## Interpretability: what the Watchdog relies on

`python watchdog/feature_ablation.py` → `results/feature_ablation.md`. On the test split, one feature group at a time is set to its training mean (0 after normalisation) and the unchanged main Watchdog is rescored. The unablated run reproduces the committed F1 of 0.772.

| Ablated group | Order F1 | Step recall | Legitimate order FPR |
|---|---|---|---|
| none | 0.772 | 0.865 | 0.043 |
| **participant: trade direction** | **0.113** | **0.018** | 0.043 |
| **all 6 participant features** | **0.001** | **0.000** | 0.008 |
| all 40 book features | 0.816 | 0.915 | 0.019 |
| book price offsets only (20) | 0.541 | 0.924 | 0.555 |
| participant: resting size / placed (each) | 0.792 / 0.814 | 0.820 / 0.821 | 0.018 / 0.031 |
| participant: order age / cancelled / side imbalance (each) | 0.766–0.773 | 0.837–0.862 | 0.039–0.047 |

- **Detection rests on the participant's own trades.** Removing trade direction alone collapses step recall to 0.018, and removing all participant features leaves nothing. Order age, cancellations and side imbalance barely matter.
- **This matches the failure on sparse spoofing.** A detector keyed on trading against a resting order struggles when that trading is sparse (SPOOFER-03 GOOG, SCRIPTED-ATK, LATEBURST-ATK).
- **Book features are not needed on the test split.** Removing all 40 slightly raises F1 and lowers false positives.
- **Caveat on the price-only row.** Its FPR of 0.555 is likely an artefact: mean prices combined with real sizes form books that never occur. The all-40 ablation, which keeps the book internally consistent, shows no such effect.
- **Scope.** Test split, one seed, single-group mean imputation; interactions between groups are not measured.

## Structural comparison with documented cases (Coscia, Sarao)

`results/real_case_comparison.md` compares order structure with the CFTC order against Panther Energy / Coscia (2013), *United States v. Coscia* (7th Cir. 2017) and the CFTC's 2015 Sarao press release. Every real-case figure was checked against those documents. Two figures from a search summary that were not in them were dropped. Simulated figures come from `python scripts/real_case_stats.py`.

| | Coscia | SPOOFER-04 INTC | SPOOFER-02 MSFT |
|---|---|---|---|
| Large orders filled | 0.08% (CME), 0.5% (ICE) | 0.4% | 1.4% |
| Large orders resting > 1 s | 0.57% | 52.2% | 43.8% |
| Reversal / side flips | algorithm operated in reverse | 73.8% of consecutive large orders | 21.3% |

- **Similar, and emergent:**
  - large orders on one side while trading on the other
  - large orders almost never filled
  - SPOOFER-04 alternating sides much like Coscia's reversing algorithm
- **Different:**
  - **Speed:** simulated large orders rest far longer, because the calibrated impact ramp rewards waiting about 50 events.
  - **Layering:** Coscia's progressively priced orders and Sarao's four to six orders held 3–4 levels back cannot emerge, because the environment allows one large order per side at the best price.
- **Not comparable:** size ratio (fixed by design), small-order fill rate, profit scale.

This is a comparison of structure, not validation of detection on real manipulation.

## Synthetic agent-based market (sim-to-sim robustness, in progress)

An evaluation-only market in which prices come from order matching between simple background traders, not from historical replay plus an impact formula. Nothing is trained in it.

- **Engine** (`synthetic/market.py`): price-time-priority limit order book.
- **Background traders** (`synthetic/simulator.py`):
  - noise traders placing limit and market orders, with every resting order cancelled at a constant per-event rate
  - imbalance followers trading toward the heavier visible side, which is how a spoof can move the price here

**Calibration (`python synthetic/calibrate.py`).** A parameter grid is scored only against real MSFT/INTC market statistics, never on spoof profitability. MSFT/INTC are used instead of all four training stocks because pooling 1-tick and 15–28-tick spreads describes no real market.

| Statistic | Real MSFT/INTC | Synthetic |
|---|---|---|
| Median spread | 1 tick | 1 tick |
| Events where the mid changes | 0.48% | 0.58% |
| Median mid range over 2,000 events | 2.0 ticks | 2.25 ticks |
| Median touch depth | 13,450 | 8,304 |
| Event mix: new / cancel / execution | 0.49 / 0.46 / 0.05 | 0.49 / 0.41 / 0.10 |

Two earlier versions failed and are recorded in the code:
- A fixed cancel probability let liquidity grow without bound (touch depth 88,770 shares, the mid never moved).
- Thin-tailed market-order sizes left the mid near-frozen in all 36 settings of the first grid. Heavier-tailed sizes fixed it.

**Emergent spoof impact (`python synthetic/spoof_impact.py`, 300 paired trials).** A 10× touch-depth spoof buy is compared with an identical market without it:

| Events after placement | Synthetic, in spreads | Replay model MSFT / INTC, in spreads |
|---|---|---|
| 10 | +0.04 ± 0.03 | +1.45 / +1.37 |
| 50 | +0.14 ± 0.07 | +2.27 / +2.12 |
| 200 | +0.55 ± 0.15 | +2.55 / +2.47 |

- **The spoof does move the synthetic price, and the effect builds up over time**, the same shape as the replay model's ramp, with no formula involved.
- **It is about 4.5× smaller at 200 events** and about 35× smaller at 10.
- **Caveat:** the size depends on the share of imbalance followers, which none of the calibration statistics constrain (fixed at 0.1). Sensitivity to that parameter is the next check.
- **Still to do:** running the frozen Spoofers and the Watchdog inside this market.

## Where the three claims stand

1. **RL learns meaningful manipulation — partly shown.** SPOOFER-04 (INTC) and SPOOFER-02 (MSFT) learned profitable spoof-and-trade behaviour. Their PnL comes from spoof gain, and the same trading with impact switched off loses money. On AAPL, GOOG and AMZN the calibrated impact makes spoofing unprofitable or marginal, and those Spoofers barely trade. Getting there required four environment fixes, each caught by breaking down where profit came from. Structurally, the learned behaviour resembles Coscia's documented pattern in the dimensions the agents choose (large orders on one side while trading on the other, fills of 0.4–1.4% vs Coscia's 0.08–0.5%, SPOOFER-04 alternating sides). It differs in speed and cannot reproduce layering (`results/real_case_comparison.md`).
2. **The Watchdog generalises to attacks it hasn't seen — not shown, now on a clean pre-registered test.** On LATEBURST-ATK, defined and committed before any of its data existed, it catches **24%** of manipulative orders against the rule's **89%**. It flags 5% of legitimate large orders against the rule's 76%. The earlier scripted-attacker result (15–25% vs 94–95%) agrees, but it was inspected during analysis and is no longer a clean test. The same weakness appears inside training: SPOOFER-03 (GOOG) spoofs on 21% of its test steps, but the Watchdog flags only 1.8%. The Watchdog learned the dense MSFT/INTC style (trading against the resting order almost every step) and misses sparser spoofing (a wait, then a few trades). It was not tuned on held-out data to hide this. The held-out RL test (SPOOFER-05, AMZN) has no manipulation to catch.
3. **The Watchdog doesn't flag legitimate activity — shown, in simulation.** It wrongly flags 1–7% of legitimate large orders, against the rule's 77–97%. It has not yet been run on real unlabeled order flow (SPY); only the rule has.

Headline comparison, Watchdog vs rule. Watchdog values are the mean ± 95% Student-t CI over 3 training seeds scored on the same data (`results/multiseed.md`):

| Measure | Watchdog (3 seeds) | Rule |
|---|---|---|
| In-distribution order F1 | **0.753 ± 0.058** | 0.352 |
| False-positive rate on legitimate orders | **0.031 ± 0.067** | 0.823 |
| False-positive rate on held-out AMZN legitimate orders | **0.004 ± 0.018** | 0.972 |
| Recall on the pre-registered unseen attacker | 0.217 ± 0.156 | **0.885** |
| Precision on the pre-registered unseen attacker | **0.774 ± 0.486** | 0.312 |
| PnL suppression of SPOOFER-04 INTC (main Watchdog, 50 episodes) | **83%** (+$12,625 → +$2,165) | 55% (→ +$5,655) |
| Interventions per episode on legitimate FLICKER orders (main Watchdog) | **0.3–0.4** | 4.7 |

- All three comparisons hold for every seed: in-distribution F1 about 2× the rule, false positives about 20× lower, and unseen-attacker recall about 4× lower.
- Seeds differ mainly in the precision/recall balance. One seed (main_s3) is conservative: order precision 0.95, recall 0.59, and step recall 0.50 versus 0.87 for the other two. Recall intervals are therefore wide (order ±0.27, step ±0.52).
- The pre-registered LATEBURST verdict was fixed in advance on the `main` model (recall ratio 1.50). The extra seeds only measure variability.
- **Basic eval on the final population (20 episodes per run, sampled actions; `results/basic_eval.md`):**
  - SPOOFER-04 INTC **+$13,208 ± $1,654** (97% of its orders manipulative) and SPOOFER-02 MSFT **+$2,437 ± $1,068** (70%). Both CIs exclude zero.
  - SPOOFER-03 GOOG −$19 ± $244 and SPOOFER-01 AAPL +$61 ± $113 are indistinguishable from zero.
  - The patient scripted attacker makes +$951 on MSFT and +$629 on INTC, and loses on AAPL, GOOG and AMZN.
  - **The rule-based baseline is weak on the RL population.** Tuned to size ≥2× depth and cancel within 200 events, it scores precision **0.20**, recall 0.82 and false-positive rate **0.84** on the RL training pool.
    - The Spoofers place many large orders they never trade against (SPOOFER-01 52 per episode, SPOOFER-05 46), and a size-plus-cancel rule can't tell those from spoofs.
    - It flags 93% of held-out SPOOFER-05's orders, every one of them legitimate.
    - On the scripted attacker it scores precision 0.56–0.57 and recall 0.93–0.97, the same on training stocks as on AMZN.
- **Basic eval on the v3 Spoofers (20 episodes per run):**
  - All five RL Spoofers make 0 trades per episode, so the RL training pool has no manipulative orders.
  - The original immediate-trading scripted attacker lost on every stock, from −$802 (AMZN) to −$5,900 (MSFT).
  - All three replayed exploit policies lose on average (−$6.9k to −$24.3k), so the fixes hold.
- **The scripted attacker now waits 30–80 market events before trading.** Over 6 episodes at 1 event per step: INTC **+$2,827** (100% of episodes profitable), MSFT **+$2,145** (83%), AAPL −$584 (0%).
- **Profit is not guaranteed to come from manipulation.** Checking where PnL comes from (spoof gain vs spread cost vs self-impact) turned out to be essential; headline PnL alone hid an env bug twice.

## Open issues (to decide before the Watchdog)

1. ~~Alternating-side spoofing very profitable on 1-tick stocks~~ — addressed by change 11. Check the retrained Spoofers in `results/basic_eval.md` against the scripted attacker to judge whether their profit is now realistic.
2. **Spoofing is unprofitable on wide-spread stocks under calibrated impact** (AAPL, GOOG, AMZN shifts are 0.37–0.62 spreads). That thins out the Spoofer population and the held-out AMZN test.
3. **The manipulation label is a proxy** (trading on the opposite side while the order rests). It stands in for intent and is not a legal determination.
4. **Seeds.** The Watchdog headline metrics use 3 training seeds with Student-t 95% CIs (`results/multiseed.md`). Each Spoofer is still a single training seed, and the basic eval's PnL CIs use a normal approximation over episodes.

## Known limitations

- Single trading day (2012-06-21) — the sample release has no other days.
- **SPY covers only 09:30–10:30**, so the planned false-positive test will use one hour of data.
- The impact layer is a model on top of historical replay; historical prices never react to the agent.
- Run-over uses "price moves through the level", not queue position.

## Design changes to sync into the Claude Project "RL"

`project-design-consolidated.md` and `execution-plan-10.md` could not be reached from this environment. Copy changes 1–10, Open issues 1–2 and the SPY one-hour limitation into both, then search for stale references ("RLlib", "10,000 shares", "ticks", "PnL = Cash + Inventory × Mid_Price", "full day" for SPY).
