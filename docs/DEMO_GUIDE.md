# 10-Minute Demo Guide

A rehearsable run sheet. Everything below is read from committed files — nothing trains, simulates or fetches
anything live.

## Before you start

```bash
streamlit run app.py --server.fileWatcherType none
```

The flag is **required**: the repo package `watchdog/` shadows the PyPI `watchdog` library Streamlit uses, and
without it the app fails on import. If the app is already open, press `R` to rerun.

**Fallback if anything breaks:** open `dashboard.html` — a static one-page summary built from the same result
files, no server needed. Keep it open in a second tab.

---

## The run sheet

### 0:00–1:00 — Problem
*Overview tab.* Read the one-sentence framing at the top. Spoofing: a large order to tilt the book, cancelled
after trading the other way. Illegal under CEA §4c(a)(5)(C). Exchanges mostly police it with static rules.

> "The question was whether a learning agent could do better. The more interesting answer turned out to be
> about how we were asking it."

### 1:00–2:00 — Research question and architecture
Two populations in a limit-order-book simulation on real LOBSTER data: PPO Spoofers rewarded only for profit,
and a recurrent Watchdog that sees one trader's activity and flags manipulation. A size-and-cancel rule is the
bar to beat.

### 2:00–4:00 — Simulation and the attacker result
*Overview KPIs.* Spoofing **emerged without instruction** on the two narrow-spread stocks: INTC
+$14,368 ± 1,135 per episode across 3 seeds. Point at the control: with the impact model switched off the same
trading loses $8,279, so the profit is the manipulation.

> "Say the caveat here, not later: this holds under our replay impact model. In a market where price emerges
> from order matching, every manipulator loses money."

### 4:00–6:00 — Demonstration
*Demonstration tab.* Step through one recorded episode: the order-book ladder, the large resting order, the
owner trading the other way, the flag, the P&L. Call it a **recorded replay**, which is what it is.

Show one legitimate episode too, so the false-positive side is visible.

### 6:00–8:00 — Results and baselines
*Detector generations tab.* The arc in three columns:

| | `main` | `causal_g0_1m` | `cov` |
|---|---|---|---|
| LATEBURST-ATK recall | 0.217 | 0.484 | **0.984** |

> "The first Watchdog failed on unseen attacks. The cause was not the learner — it was the label. We were
> marking every step an order rested, including steps before any opposite trade, so the model learned to
> predict *who would spoof* rather than detect *the act*. It flagged 99% of one spoofer's orders before any
> trade happened."

That is the strongest minute in the talk. Land it.

### 8:00–9:00 — Leakage-resistant evaluation and limitations
*Is the task hard? tab.* Then the honest bound:

> "We then checked whether we had actually built a detector. The label is 'the owner traded the opposite side
> while the order rested' — and two of the model's own input features are the trade direction and the resting
> side. Comparing them with no parameters at all scores 0.997. We had reached a ceiling that a rule with no
> parameters already occupied."

> "So we removed the attribution entirely and used only the public order book. Step-level F1 drops from 0.95
> to **0.392** — still better than a constant classifier's 0.218, so it's a real signal, but that is the
> honest size of the problem."

### 9:00–10:00 — Contribution and future work
Three sentences:
1. RL agents discovered profitable manipulation unprompted, with a control proving the profit is the spoof.
2. A labelling choice silently turned a detection task into an actor-identification task; finding and fixing
   that moved held-out recall from 0.22 to 0.98 across four pre-registered attack families.
3. The detector's apparent success is a ceiling effect, and the genuinely hard problem — attributing orders
   and trades to accounts on an anonymised feed — is the next piece of work.

---

## Questions you should expect

| Question | Answer |
|---|---|
| "Does this detect real spoofing?" | No, and we don't claim it. LOBSTER has no participant identities, so the detector cannot even be run on real order flow — only the rule can, as a flag rate with no labels. |
| "Why is RL in the title if a supervised LSTM matches it?" | It shouldn't be, for the detector. RL earns its place on the attacker side, where there is no label to supervise against. We say so in the report. |
| "Isn't 0.98 recall suspiciously high?" | Yes — that's why we checked. It's a ceiling a zero-parameter rule also reaches. `docs/MODEL_COMPARISON.md` §1. |
| "How do we know the pre-registered tests are genuine?" | Attacker, control and hypotheses were committed before the data existed; git history shows the ordering; each ran once. The first two went against us. |
| "Did the train/test overlap invalidate things?" | It was a real defect — 33 of 50 test episodes overlapped training. Measured cost: 0.031 F1, below seed noise. Both protocols are committed. |
| "How much is seed variance?" | Material. MSFT Spoofer P&L spans +$3,628 to +$15,886. We report intervals, not just means. |

---

## What not to say

- Don't call the demo "live" or "real-time". It replays recorded episodes.
- Don't quote the attribution-free **episode-level F1** (0.753) — a constant classifier scores 0.769.
- Don't present replay-model profitability as real-market profitability.
- Don't describe the environment as a realistic exchange simulator: no queue position, no fees, no latency,
  and historical prices never react to the agent.
