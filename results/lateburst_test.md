# Pre-registered held-out test: LATEBURST-ATK

Defined and committed before any of its data or results existed; nothing was changed after seeing them.

**Hypothesis:** Watchdog order-level recall on MSFT+INTC >= 2x recall on AAPL+GOOG+AMZN.

**Result: not supported.** Watchdog order-level recall MSFT+INTC = 0.296, AAPL+GOOG+AMZN = 0.198, ratio = 1.50.

Rule tuned on the train split: size ≥ 2× depth, cancelled within 1000 events. 50 episodes per stock of LATEBURST-ATK plus FLICKER negatives.

| Group | Manipulative / orders | Watchdog P | Watchdog R | Watchdog FPR | Rule P | Rule R | Rule FPR | Median delay W / R |
|---|---|---|---|---|---|---|---|---|
| All stocks | 1132 / 4044 | 0.66 | 0.24 | 0.05 | 0.31 | 0.89 | 0.76 | 12 / 22 |
| MSFT+INTC | 479 / 1566 | 0.75 | 0.30 | 0.04 | 0.33 | 0.90 | 0.81 | 0 / 22 |
| AAPL+GOOG+AMZN | 653 / 2478 | 0.58 | 0.20 | 0.05 | 0.30 | 0.88 | 0.73 | 16 / 22 |
| AAPL | 206 / 849 | 0.54 | 0.25 | 0.07 | 0.28 | 0.86 | 0.70 | 17 / 22 |
| MSFT | 236 / 785 | 0.77 | 0.29 | 0.04 | 0.32 | 0.88 | 0.79 | 0 / 22 |
| GOOG | 214 / 831 | 0.63 | 0.16 | 0.03 | 0.30 | 0.85 | 0.68 | 16 / 22 |
| INTC | 243 / 781 | 0.74 | 0.30 | 0.05 | 0.33 | 0.91 | 0.83 | 0 / 22 |
| AMZN | 233 / 798 | 0.59 | 0.19 | 0.05 | 0.32 | 0.91 | 0.80 | 16 / 22 |

Attacker PnL per episode (context only): AAPL -471, MSFT +442, GOOG -735, INTC +726, AMZN -330
