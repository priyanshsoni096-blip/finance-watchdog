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
| Watchdog: dataset recorder, env, RecurrentPPO training, evaluation vs baseline | Code done and tested (smoke-run end to end); full training run pending |
| Coscia-style layering attacker, synthetic market, SPY false-positive test, real-case comparison, multi-seed CIs | Not started |

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
64 tests, about 1–2 minutes. Tests that need data skip themselves if it isn't downloaded.

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
python watchdog/dataset.py --episodes 30
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

### Reading `results/basic_eval.md`

- **C1 table** — one row per agent run. Compare a trained Spoofer's PnL with HONEST and SCRIPTED-ATK on the same stock. "Trades against own spoof/ep" is the spoofing signature (spoof one side, trade the other). "PnL (spread·lots)" makes stocks comparable. Look at the CI: if it crosses zero, the profit isn't established.
- **Detector table** — the rule-based baseline, tuned on training stocks only, scored per bucket. "Manipulative" orders are ones whose owner traded on the opposite side while they rested. The held-out AMZN rows are the generalisation check. The "Flicker only" row is the false-positive check on large-order-then-cancel with no trading.
- **Real-flow table** — the same rule on real LOBSTER messages. There are no labels, so this is a flag *rate*, not a false-positive rate.

---

## What is what

```
data/download.py            fetch raw LOBSTER CSVs from Hugging Face
env/lobster_data.py         parse message + orderbook CSVs into arrays (LobsterDay), cache
env/normalization.py        scale-free features: prices in spread units, sizes as log ratio to trailing size
env/price_impact.py         linear/sqrt impact; lambda fitted from order-flow imbalance
env/lob_env.py              LimitOrderBookEnv (Gymnasium): actions, fills, run-over, reward
scripts/calibrate.py        per-stock statistics -> configs/calibration.json
training/train_spoofer.py   PPO training for the SPOOFER roster, behaviour logging
training/summarize.py       training trend table from progress.csv
evaluation/baseline_detector.py   rule-based detector, metrics, tuning, real-data front-end
evaluation/basic_eval.py    the basic evaluation described above
tests/                      64 tests (data, normalization, impact, env, detector)
checkpoints/                trained models + logs (git-ignored)
results/                    evaluation outputs
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
- **Profit is not guaranteed to come from manipulation.** Checking where PnL comes from (spoof gain vs spread cost vs self-impact) turned out to be essential; headline PnL alone hid an env bug twice.

## Open issues (to decide before the Watchdog)

1. ~~Alternating-side spoofing very profitable on 1-tick stocks~~ — addressed by change 11. Check the retrained Spoofers in `results/basic_eval.md` against the scripted attacker to judge whether their profit is now realistic.
2. **Spoofing is unprofitable on wide-spread stocks under calibrated impact** (AAPL, GOOG, AMZN shifts are 0.37–0.62 spreads). That thins out the Spoofer population and the held-out AMZN test.
3. **The manipulation label is a proxy** (trading on the opposite side while the order rests). It stands in for intent and is not a legal determination.
4. The basic eval uses one seed per agent and a normal-approximation CI.

## Known limitations

- Single trading day (2012-06-21) — the sample release has no other days.
- **SPY covers only 09:30–10:30**, so the planned false-positive test will use one hour of data.
- The impact layer is a model on top of historical replay; historical prices never react to the agent.
- Run-over uses "price moves through the level", not queue position.

## Design changes to sync into the Claude Project "RL"

`project-design-consolidated.md` and `execution-plan-10.md` could not be reached from this environment. Copy changes 1–10, Open issues 1–2 and the SPY one-hour limitation into both, then search for stale references ("RLlib", "10,000 shares", "ticks", "PnL = Cash + Inventory × Mid_Price", "full day" for SPY).
