# Model Comparison

Every detector in the project on comparable data, with the experiment that produced each number. Protocol in
`docs/EXPERIMENT_PROTOCOL.md`; audit in `docs/FINAL_AUDIT.md`.

**Read §1 before §2.** Section 2's numbers are bounded by section 1's ceiling.

---

## 1. The ceiling: what the label itself is worth

The label is `opposite_trades > 0` — the owner traded the opposite side while the order rested. Observation
feature 44 is the participant's trade direction, feature 45 the side its resting orders sit on. Comparing
their signs reproduces the label's definition, with **no parameters and no fitting**.

| Detector | In-distribution order F1 | Held-out scripted order F1 | Legitimate FPR | Experiment |
|---|---|---|---|---|
| **Zero-parameter signature** | **0.997** | **0.995** | 0.000 | `results/label_observability.json` |
| LABEL-RULE (same idea, Arnav) | 0.986 | 0.995 | 0.000 | `results/baselines_final.json` |
| "A resting order is present" | 0.362 | — | 1.000 | `results/label_observability.json` |

The third row matters: mere presence of a large order does **not** identify manipulation. It is specifically
the two-feature comparison that carries the label.

---

## 2. Attributed detectors — all given features 44 and 45

### 2a. Order-level recall / legitimate FPR, 3 seeds

Source `results/baselines_final.json`. Same observation, same causal labels, thresholds fitted on train only.

| Dataset | Label ceiling | SUP-LSTM | RL Watchdog `cov` | LOGREG | Size+cancel rule |
|---|---|---|---|---|---|
| Original test split | 0.995 / 0.000 | 0.995 / 0.000 | 0.995 / 0.002 | 0.469 / 0.123 | 0.941 / 0.812 |
| Held-out (SCRIPTED, AMZN) | 0.990 / 0.000 | 0.990 / 0.000 | 0.990 / 0.004 | 0.198 / 0.000 | 0.940 / 0.869 |
| LATEBURST-ATK | 0.983 / 0.000 | 0.983 / 0.000 | 0.984 / 0.002 | 0.218 / 0.098 | 0.885 / 0.819 |
| DRIP-ATK | 0.989 / 0.000 | 0.989 / 0.000 | 0.989 / 0.012 | 0.324 / 0.144 | 0.744 / 0.797 |
| REVERSAL-ATK | 0.984 / 0.000 | 0.984 / 0.001 | 0.983 / 0.012 | 0.218 / 0.099 | 0.964 / 0.804 |

**A supervised LSTM matches the RL detector everywhere and trains in minutes.** Logistic regression fails
(0.20–0.47), which shows the task needs the *interaction* between trade side and resting side — a two-feature
interaction, which is §1 restated.

### 2b. Three Watchdog generations

Source `results/posthoc_rescore.json`, order level, 3 seeds, mean ± 95% Student-t CI. Post-hoc except where a
test was pre-registered for that model.

| Metric | `main` | `causal_g0_1m` | `cov` | Rule |
|---|---|---|---|---|
| In-distribution order F1 | 0.753 ± 0.058 | 0.978 ± 0.015 | **0.984 ± 0.001** | 0.352 |
| Legitimate order FPR | 0.031 ± 0.067 | 0.003 ± 0.012 | **0.002 ± 0.002** | 0.820 |
| Held-out AMZN legitimate FPR | 0.004 ± 0.018 | 0.001 ± 0.004 | **0.001 ± 0.002** | 0.972 |
| SCRIPTED-ATK train stocks recall | 0.257 ± 0.199 | 0.504 ± 0.024 | **0.991 ± 0.001** | 0.938 |
| LATEBURST-ATK recall | 0.217 ± 0.156 | 0.484 ± 0.022 | **0.984 ± 0.001** | 0.885 |
| DRIP-ATK recall | 0.067 ± 0.115 | 0.387 ± 0.273 | **0.989 ± 0.002** | 0.744 |
| REVERSAL-ATK recall | 0.139 ± 0.129 | 0.451 ± 0.110 | **0.983 ± 0.005** | 0.964 |

The jump from `main` to `causal_g0_1m` is the **causal label** and γ = 0; from there to `cov` is **training
coverage of varied act timing**. Neither is a hyperparameter search.

### 2c. Independent cross-platform verification

`cov` was trained on macOS. Re-scored on Windows from the committed checkpoint: in-distribution F1 **0.984**,
legitimate FPR **0.001**, SCRIPTED train-stocks recall **0.991**, SCRIPTED AMZN recall **0.983** — matching the
committed values to three decimals. The DRIP and REVERSAL datasets were rebuilt from scratch and their
pre-registered tests re-run: 1,164 and 1,615 numeric values, **0 real differences** (five per-stock P&L figures
differ at relative ~1e-15).

---

## 3. Detection WITHOUT attribution

Source `results/anonymised_experiment.json`. The whole participant block is withheld; only the displayed order
book and aggregates derived from it are used. Temporal splits; thresholds fitted on train only.

**Leakage audit of what remains:** withheld pair step F1 0.948 · best retained single-feature rule 0.490
(`total displayed depth`) · largest retained correlation 0.381 (`book L1 bid size`) · retained features with
correlation > 0.5: **0**. The removal is real.

3 seeds, mean ± 95% Student-t CI. ±0.000 means the model is deterministic given the data (the constants,
the fitted rule, and logistic regression under lbfgs); only gradient boosting and the LSTM vary with the seed.

### Step level — is manipulation happening now?

| Model | Held-out precision | Recall | F1 | FPR | PR-AUC |
|---|---|---|---|---|---|
| constant: never flag | n/a | 0.000 | 0.000 | 0.000 | 0.096 |
| constant: always flag | 0.122 | 1.000 | 0.218 | 1.000 | 0.096 |
| book-imbalance rule | 0.120 | 0.085 | 0.099 | 0.086 | 0.100 |
| **logistic regression** | 0.409 | 0.376 | **0.392** | 0.075 | 0.314 |
| gradient boosting | 0.540 ± 0.016 | 0.125 ± 0.015 | 0.203 ± 0.020 | 0.015 ± 0.002 | **0.321 ± 0.002** |
| supervised LSTM | 0.311 ± 0.023 | 0.028 ± 0.007 | 0.052 ± 0.011 | 0.009 ± 0.002 | 0.166 ± 0.015 |

**Logistic regression wins on F1 against the constant (0.392 vs 0.218), and PR-AUC is about 3× the base
rate (0.32 vs 0.096).** There is real signal in the book alone — modest, but not nothing.

**In-distribution, for contrast:** the LSTM reaches F1 0.864 ± 0.020 and PR-AUC 0.933 ± 0.026, far ahead of
everything else, then collapses out of distribution. A sequence model memorises the training sources'
temporal signatures. Had only in-distribution been reported, it would have looked like the clear winner.

### Episode level — does this window contain manipulation?

| Model | Held-out precision | Recall | F1 | FPR |
|---|---|---|---|---|
| constant: always flag | 0.625 | 1.000 | **0.769** | 1.000 |
| logistic regression | 0.685 | 0.836 | 0.753 | 0.640 |
| gradient boosting | 0.975 ± 0.013 | 0.416 ± 0.010 | 0.583 ± 0.010 | **0.018 ± 0.010** |
| supervised LSTM | 0.732 ± 0.048 | 0.288 ± 0.034 | 0.413 ± 0.043 | 0.176 ± 0.025 |

**Episode-level F1 is not a result.** The held-out set is 62.5% positive, so flagging everything beats every
model. Do not quote it. Gradient boosting is the only model that separates anything usefully here, and it does
so by being conservative — precision 0.975 at FPR 0.018 — not by winning on F1.

### Per instrument — the finding the pooled number hides

Step-level F1 per source, held-out, seed 0. "Always flag" scores 0.30–0.36 on each attacker source.

| Source | Logistic regression | Gradient boosting | Supervised LSTM | Constant |
|---|---|---|---|---|
| SCRIPTED-ATK_INTC | **0.585** | 0.433 | 0.083 | 0.346 |
| SCRIPTED-ATK_MSFT | **0.600** | 0.297 | 0.098 | 0.356 |
| SCRIPTED-ATK_AAPL | 0.073 | 0.000 | 0.004 | **0.300** |
| SCRIPTED-ATK_GOOG | 0.126 | 0.011 | 0.007 | **0.300** |
| SCRIPTED-ATK_AMZN | 0.131 | 0.006 | 0.022 | **0.333** |

**Attribution-free detection works on the two 1-tick stocks and loses to a constant on the other three.**
Those are exactly the stocks where spoofing is profitable, which is coherent — but it means the pooled 0.392
averages two opposite results and describes neither.

**Why is unresolved.** Two mechanical explanations were tested and both refuted
(`results/detectability_by_instrument.md`): the attacker places the same 10× touch depth on every stock, and
the shift it produces in displayed book imbalance is *smaller* on the two detectable stocks, with
signal-to-noise failing to rank with detectability (correlation −0.06). What remains is a descriptive
association with spread width across **five** instruments, which cannot support a mechanism. Tick-size regime
and spread width coincide in this dataset and cannot be separated.

### Calibration (Brier, lower is better)

A classifier predicting the base rate scores **0.099**.

| Model | In-distribution | Held-out |
|---|---|---|
| logistic regression | 0.110 | 0.139 |
| gradient boosting | 0.080 | 0.110 |
| supervised LSTM | **0.029** | 0.128 |

**No model is calibrated out of distribution** — all three score worse than the base-rate constant. They rank
better than chance; their probabilities should not be read as probabilities.

---

## 3b. Does detection actually cost the attacker anything?

Re-measured with the CP-7 detector acting **inside** the episode: on a flag the exchange cancels the
participant's resting large orders and blocks new ones for 200 market events
(`results/posthoc/cov/pnl_suppression.md`, 50 episodes, same seeds for every condition).

| Source | No surveillance | CP-7 Watchdog | Suppressed | Rule | Suppressed |
|---|---|---|---|---|---|
| SPOOFER-04 (INTC) | +$12,625 ± 1,162 | **−$133 ± 190** | **101%** | +$5,655 ± 657 | 55% |
| SPOOFER-02 (MSFT) | +$3,186 ± 517 | **−$4,557 ± 1,210** | 243% | +$140 ± 831 | 96% |
| SCRIPTED-ATK (INTC) | +$748 ± 634 | −$2,458 ± 496 | 429% | −$1,281 ± 613 | 271% |
| LATEBURST-ATK (MSFT) | +$531 ± 674 | −$2,728 ± 430 | 614% | −$974 ± 757 | 283% |

The earlier measurement used the original Watchdog and left SPOOFER-04 at +$2,165 (83% suppressed). CP-7
drives it to a loss. Interventions against legitimate FLICKER orders: **0.0 per episode** for CP-7 against the
rule's 4.7.

Two cautions. Suppression above 100% means the attacker ends up losing money, which flatters the number —
read the P&L column, not the percentage. And **the attackers are frozen**: they do not adapt to being watched,
so this measures surveillance against non-adaptive manipulation.

## 4. The attacker side, where RL is justified

| Agent | Stock | P&L / episode (replay) | Impact off | Synthetic market |
|---|---|---|---|---|
| SPOOFER-04 | INTC | **+$14,368 ± 1,135** (3 seeds) | −$8,279 | −$3,270 ± 688 |
| SPOOFER-02 | MSFT | **+$11,431 ± 16,844** (3 seeds) | −$7,037 | −$2,930 ± 686 |
| SPOOFER-01/03/05 | AAPL/GOOG/AMZN | ~0, never learned to spoof | — | — |

The impact-off column is the control: the same trading loses money without the impact model, so the profit is
the manipulation. MSFT's interval spans +$3,628 to +$15,886 across seeds and is reported, not smoothed.
Profitability **does not survive** a change of market mechanism.

Here there is no label to supervise against and the agent must find the strategy from profit alone. **This is
where RL earns its place in the project; the detector is not.**

---

## 5. Recommendation

| Use | Model | Why |
|---|---|---|
| Detection, task as posed | **SUP-LSTM or gradient boosting** | matches the RL detector, trains in minutes, far simpler |
| Detection, attribution-free | **logistic regression** (step level) | the only model that beats a constant; GB if low FPR matters more |
| Attack discovery | **PPO** | no labels exist to supervise against |
| Reported as a contribution | the **causal-label diagnosis** and the attacker side | the 0.98 recall is a ceiling effect |
