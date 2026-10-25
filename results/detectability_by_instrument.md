# Why detection works on some instruments and not others

With participant attribution withheld, step-level detection splits sharply by instrument. The scripted attacker is the same program on every stock, so the difference belongs to the market. Two mechanical explanations were tested. **Both are refuted.**

| Stock | Spread (ticks) | Touch depth | Spoof / depth | Book shift | Book noise (sd) | Signal/noise | Step F1 |
|---|---|---|---|---|---|---|---|
| MSFT | 1 | 13,318 | 10.00 | 0.0051 | 0.0951 | 0.054 | **0.600** |
| INTC | 1 | 13,915 | 10.00 | 0.0099 | 0.0968 | 0.103 | **0.585** |
| AMZN | 13 | 164 | 9.99 | 0.0275 | 0.1658 | 0.166 | **0.131** |
| GOOG | 28 | 138 | 10.00 | 0.0034 | 0.1619 | 0.021 | **0.126** |
| AAPL | 15 | 150 | 10.01 | 0.0126 | 0.1794 | 0.070 | **0.073** |

## H1 — the spoof is proportionally larger where it is detected. **Refuted.**

The attacker places the same multiple of touch depth on every stock (10.01x, by construction), so attack size cannot explain any difference.

## H2 — the spoof moves the book more, or further relative to its noise. **Refuted.**

Correlation of the book-imbalance shift with step F1: **-0.384**. Correlation of signal-to-noise with step F1: **-0.057**. Neither explains the split, and the shift is in fact *smaller* on the two detectable stocks than on AAPL and AMZN. The signal-to-noise ranking disagrees with the detectability ranking outright.

## What is left

A descriptive association with spread width (correlation **-0.842** with step F1) on **five instruments**. That is far too small a sample to support a mechanism, and none is claimed here. The per-instrument split is a reproducible observation without an established cause.

Two further possibilities were *not* tested and would need more instruments or more trading days: that the effect is multivariate across the 153 features rather than present in any single order-book statistic, and that it tracks tick-size regime (both detectable stocks are 1-tick) rather than spread width as such — which this dataset cannot separate, because the two coincide.

