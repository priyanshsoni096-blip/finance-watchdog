# RL Finance Watchdog

**An Adversarial Reinforcement Learning System for the Detection of Limit Order Book Spoofing**

> Can a reinforcement learning agent (the "Watchdog") learn to detect *previously unseen* order-book manipulation strategies — specifically spoofing — while maintaining a low false-positive rate on legitimate market activity, without being given explicit rules about what spoofing looks like?

This project demonstrates the feasibility of an RL-based surveillance approach in a realistic limit-order-book simulation, and evaluates the structural similarity of the emergently-learned manipulation strategy to documented real-world manipulation cases.

No labeled real-world spoofing dataset exists to validate detection recall against, so this project makes no claim of real-world detection capability.

Legal reference: CEA §4c(a)(5)(C) (Dodd-Frank §747).

## Status

| Phase | State |
|---|---|
| 1. Environment, price impact, cross-stock normalization | **Done** — 51 tests pass on real data |
| 2. Spoofer population (SPOOFER-01..05) | Not started |
| 3+. Watchdog, rule-based baseline, evaluation matrix | Not started |

## Setup

```bash
pip install -r requirements.txt
python data/download.py          # ~600 MB: AAPL, MSFT, GOOG, INTC, AMZN at 10 levels
python scripts/calibrate.py      # writes configs/calibration.json
python -m pytest tests -q
```

Data: LOBSTER sample release (2012-06-21) from Hugging Face `totalorganfailure/lobster-data`, fetched as raw CSVs.

## Environment (`env/lob_env.py`)

- **Actions** `Discrete(6)`: no-op, market buy 1 lot, market sell 1 lot, spoof buy at best bid, spoof sell at best ask, cancel spoofs.
- **Lot** = half the stock's median touch depth, rounded to 100 shares (AAPL/GOOG/AMZN 100, MSFT 6,700, INTC 7,000). A spoof is 10 × trailing touch depth, a median of 14–20 lots on every ticker.
- **Episodes** are 2,000 events and never terminate early. At the end, any remaining position is liquidated across the spread.
- **Observation** `Box(43)`: 40 book features (10 levels × ask/bid price and size, normalized) + inventory, PnL, active spoof size (all scale-free). Resting spoofs are added to the displayed size at their price level.
- **Fills** happen at the *impacted* touch; **inventory is marked** to the *unimpacted* historical mid.
- **Execution risk**: a resting spoof is run over (filled) when the historical opposite touch crosses its price.
- **Reward**: PnL change in units of (trailing spread × lot) minus `inv_penalty × |inventory| / lot`.

## Measured calibration (`python scripts/calibrate.py`)

| Ticker | Events | Median mid | Median spread | Spread (ticks) | Touch depth | OFI λ | OFI R² | 10×-depth spoof moves mid by |
|---|---|---|---|---|---|---|---|---|
| AAPL | 400,391 | $583.24 | $0.15 | 15 | 150 | 0.046 | 0.40 | 0.46 spreads |
| MSFT | 668,765 | $30.51 | $0.01 | 1 | 13,318 | 0.255 | 0.73 | 2.55 spreads |
| GOOG | 147,916 | $569.88 | $0.27 | 28 | 138 | 0.062 | 0.46 | 0.62 spreads |
| INTC | 624,040 | $27.02 | $0.01 | 1 | 13,915 | 0.247 | 0.66 | 2.47 spreads |
| AMZN | 269,748 | $222.66 | $0.13 | 13 | 164 | 0.037 | 0.33 | 0.37 spreads |

Normalized best-level price feature, median |value|: AAPL 0.500, INTC 0.500. Best-level size feature, median: AAPL 0.521, INTC 0.673.

## Design changes from the handoff, and why

Each of these was driven by a measurement, not preference.

1. **Stable-Baselines3 / sb3-contrib instead of Ray RLlib.** Development machine is a Windows CPU laptop (i7-1165G7, 16 GB, no GPU, Python 3.13), where Ray support is beta. Architecture and hyperparameters are unchanged; the Watchdog will use `RecurrentPPO` (LSTM).
2. **Prices normalized in spread units, not ticks.** Median spread is 15 ticks on AAPL and 1 tick on INTC, so tick offsets are not comparable across stocks. In spread units both give a median best-level offset of 0.500.
3. **Spoof size = 10 × trailing touch depth, not a fixed 10,000 shares.** Touch depth ranges from ~140 shares (GOOG) to ~13,900 (INTC); a fixed share count means wildly different things per stock.
4. **Impact λ fitted per ticker from order-flow imbalance** (Cont, Kukanov & Stoikov 2014), not from order-size percentiles. The percentile rule (95th-percentile order moves the mid by the median move) was implemented and **rejected**: on MSFT and INTC it implies a 10×-depth spoof moves the price 44–58 spreads, because typical orders there are ~0.09× touch depth. It is kept in `env/price_impact.py` and a test records the failure. λ varies ~7× across tickers even under OFI, so a single pooled λ is not used.
5. **Inventory marked to the unimpacted mid.** With the handoff's `Cash + Inventory × Mid_Price` using the impacted mid, an agent could earn reward just by holding a position while a spoof rests, without ever trading at the distorted price. Now the only profitable pattern is spoof → trade at the distorted price → cancel. Tested: holding through a spoof produces an identical reward stream with impact on or off; spoof-then-sell gains exactly `lot × shift` ($12.76 AAPL, $4.98 INTC in the test windows).
6. **Reward in scale-free units.** Dollar PnL and a per-share penalty are not comparable between a $583 and a $27 stock.
7. **Lot size scaled to depth; no early termination.** A first version used 100-share lots, a 10-lot inventory limit and terminated when the limit was breached. Measured over 40,000 random steps per ticker, a spoof was 14–16 lots on AAPL/GOOG/AMZN and 1,333–1,396 lots on MSFT/INTC, so every run-over breached the limit. **All 31 of 31 early terminations happened on a run-over step.** That gives the agent a way to end a losing episode on purpose. After the fix: 0 terminations, and spoofs of 14–20 lots everywhere.
8. **End-of-episode liquidation.** Marking to mid at the end would make holding inventory free; the position is instead closed at the bid (long) or ask (short).
9. **30,000-event warmup** before an episode may start. The opening book is thin: INTC's trailing touch depth is 0.08× its day median at event 15,000. With a 5,000-event warmup, 3.9% of INTC starts produced spoofs under 5 lots; with 30,000, none on any ticker did.

## Known limitations

- Single trading day (2012-06-21) for all tickers — the sample release has no other days.
- **SPY covers only 09:30–10:30** (file window `34200000_37800000`), so the false-positive test will use one hour of data.
- The price-impact layer is a model on top of historical replay; historical prices themselves never react to the agent.
- Run-over uses a "price moves through the level" rule, not queue position.

## Design changes to sync into the Claude Project "RL"

`project-design-consolidated.md` and `execution-plan-10.md` could not be reached from this environment. Items 1–6 above, and the SPY one-hour limitation, need to be copied into both documents, then searched for stale references ("RLlib", "10,000 shares", "ticks", "PnL = Cash + Inventory × Mid_Price", "full day" for SPY).
