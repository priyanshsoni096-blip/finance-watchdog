# Feature Dictionary

Every input any detector in this project receives, what it means, when it becomes available, and whether it
leaks the target. Generated from `evaluation/rollout.py`, `env/normalization.py` and
`watchdog/anonymised_experiment.py`; the leakage column is measured, not asserted — see
`results/label_observability.md` and `tests/test_no_attribution_leakage.py`.

---

## The target being predicted

| Label | Definition | Availability |
|---|---|---|
| `manipulative` (order level) | `opposite_trades > 0` — the order's owner traded the opposite side while it rested | retrospective, known only after the order is removed |
| **hindsight** (step level) | every step a manipulative order rested | **attaches the outcome to steps before the evidence exists** |
| **causal** (step level, current) | from the first step the owner trades opposite a resting order until removal | available at the step it fires |

Both are proxies for intent, not legal determinations.

---

## Observation: 46 features

### Book block — features 0–39

Ten price levels per side, in LOBSTER column order, repeated per level
`[ask price, ask size, bid price, bid size]`.

| Index | Name | Definition | Available at decision time? | Leaks the label? |
|---|---|---|---|---|
| 0, 4, 8, … 36 | `book L{n} ask price` | (ask price − mid) / reference spread, clipped | yes | no |
| 1, 5, 9, … 37 | `book L{n} ask size` | log1p(size / trailing ask reference size) | yes | no |
| 2, 6, 10, … 38 | `book L{n} bid price` | (bid price − mid) / reference spread, clipped | yes | no |
| 3, 7, 11, … 39 | `book L{n} bid size` | log1p(size / trailing bid reference size) | yes | no |

Prices are in **spread units** and sizes are **log ratios to trailing depth**, so a $585 stock and a $27 stock
are comparable. **The monitored participant's resting orders are included in the displayed size**, exactly as
a real market feed would show them — this block does not identify *whose* orders they are.

### Participant block — features 40–45 (privileged attribution)

These exist only because the simulator knows which orders and trades belong to the monitored participant. A
real anonymised feed cannot produce them.

| Index | Name | Definition | Available? | Leaks the label? |
|---|---|---|---|---|
| 40 | `resting size` | log1p(total resting size / trailing depth) | yes | partially — identifies that *a* large order rests |
| 41 | `oldest resting age` | age of the oldest resting order / 200, capped at 5 | yes | no |
| 42 | `placed this step` | log1p(size placed / depth) | yes | no |
| 43 | `cancelled this step` | log1p(size cancelled / depth) | yes | no |
| **44** | **`trade direction`** | +1 buy, −1 sell, 0 none — **this participant's trade this step** | yes | **yes, with 45** |
| **45** | **`resting imbalance`** | signed log1p(\|resting bid − resting ask\| / depth) — **which side it rests on** | yes | **yes, with 44** |

**Features 44 and 45 together reconstruct the label's definition.** `sign(45) × sign(44) < 0` is precisely
"traded the opposite side while resting". Measured, with no parameters and no fitting:

| Group | Precision | Recall | Order F1 |
|---|---|---|---|
| In-distribution (test) | 0.999 | 0.994 | **0.997** |
| Held-out scripted attacker | 1.000 | 0.990 | **0.995** |
| Legitimate only | — | — | FPR **0.000** |

This is **definitional leakage**, not temporal leakage: the information is genuinely available at the decision
step, but the target is a function of the input. A correct temporal split does not fix it.

For contrast, feature 40 alone ("a resting order is present") scores order F1 **0.362** at FPR **1.000** — the
label is carried specifically by the 44/45 interaction, not by order presence.

---

## Attribution-free feature set

Used by `watchdog/anonymised_experiment.py`. **The entire participant block is withheld.** The book block is
kept, plus eleven aggregates derived from it alone.

| Index | Name | Definition |
|---|---|---|
| 40 | `level-1 imbalance` | top bid size − top ask size |
| 41 | `level-1 imbalance (normalised)` | that, over their sum |
| 42 | `deep imbalance` | total bid size − total ask size across 10 levels |
| 43 | `deep imbalance (normalised)` | that, over their sum |
| 44 | `total displayed depth` | total ask + total bid size |
| 45 | `spread` | top ask price − top bid price, in spread units |
| 46 | `largest ask level` | max displayed size on the ask side |
| 47 | `largest bid level` | max displayed size on the bid side |
| 48 | `largest ask - largest bid` | difference of the two above |
| 49 | `ask size dispersion` | standard deviation of ask sizes across levels |
| 50 | `bid size dispersion` | standard deviation of bid sizes across levels |

> **Index collision warning.** In the attribution-free matrix, index 44 is `total displayed depth` — *not* the
> participant's trade direction. The two feature spaces are different; always say which one you mean.

Each of these 51 columns is then expanded to three: `[current value, mean of the previous 10 steps, current −
that mean]`, giving **153 columns**. The rolling window is computed **per episode**, so history never crosses
an episode boundary.

### Measured leakage audit of the retained set

| Check | Value |
|---|---|
| Withheld pair (44 + 45) step F1 — what was removed | **0.948** |
| Largest absolute correlation of any retained feature with the label | 0.381 (`book L1 bid size`) |
| Retained features correlating above 0.5 | **0** |
| Best single retained feature threshold rule, swept on train | 0.490 (`total displayed depth`) |

The removal is real: no retained column reconstructs the label by another route. Enforced by
`tests/test_no_attribution_leakage.py`, which fails if any retained column ever reproduces a withheld one or
becomes near-deterministic for the label.

---

## What is *not* in any feature set

Participant identity or account ID; any future event; the label itself or any direct transform of it; order
queue position; exchange fees; latency. The simulator does not model the last three at all.
