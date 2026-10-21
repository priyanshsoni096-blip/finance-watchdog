# Rule on real, unlabeled order flow (including SPY)

No labels exist, so these are flag **rates**, not false-positive rates. The Watchdog cannot run on real flow: LOBSTER has no participant identities, so its participant features cannot be built. Rule: size ≥ 2× trailing touch depth, deleted within M events without a visible execution first. Events after a 30,000-event warmup.

| Stock | Window | Events after warmup | New orders | Large orders | Flags, M=200 | per 10k events | Flags, M=1000 | per 10k events | Share of large orders flagged (M=1000) |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | 09:30-16:00 | 370,391 | 176,867 | 2,037 | 1,322 | 35.69 | 1,388 | 37.47 | 68% |
| MSFT | 09:30-16:00 | 638,765 | 314,831 | 12 | 2 | 0.03 | 5 | 0.08 | 42% |
| GOOG | 09:30-16:00 | 117,916 | 56,666 | 308 | 209 | 17.72 | 214 | 18.15 | 69% |
| INTC | 09:30-16:00 | 594,040 | 290,056 | 32 | 17 | 0.29 | 24 | 0.40 | 75% |
| AMZN | 09:30-16:00 | 239,748 | 117,265 | 2,921 | 2,443 | 101.90 | 2,562 | 106.86 | 88% |
| SPY | 09:30-10:30 | 1,122,693 | 557,100 | 16 | 3 | 0.03 | 6 | 0.05 | 38% |
