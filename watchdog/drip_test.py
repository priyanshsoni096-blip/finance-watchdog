"""Pre-registered clean held-out test with DRIP-ATK (CP-7; see README, "Pre-registered test: DRIP-ATK").

This script, the attacker (`DripSpoof`), the legitimate control (`SameSideTrader`) and the hypotheses were committed
before any DRIP-ATK or SAMESIDE data was generated for scoring and before any detector saw them. Run it once; do not
change any Watchdog, the rule, thresholds or features after seeing its output.

Usage:
    python watchdog/drip_test.py

Procedure (fixed in advance):
  * 50 episodes per stock (AAPL, MSFT, GOOG, INTC, AMZN) each of DRIP-ATK, SAMESIDE (legitimate: large order plus
    same-side trades) and FLICKER (legitimate: large order, no trades), on test-split seeds, 10 events per step.
  * Verdict model: the CP-7 Watchdog `checkpoints/WATCHDOG/causal_g0_1m` (causal label, gamma 0, 1M steps, frozen
    in commit ef76cb4), with its saved observation normalisation. Baseline: the rule tuned on the train split.
  * H1: Watchdog order-level F1 on DRIP-ATK + SAMESIDE + FLICKER (all five stocks pooled) >= rule F1.
  * H2: Watchdog order-level false-positive rate on SAMESIDE orders <= 0.10.
  * Context only, not used for any verdict: seeds causal_g0_1m_s2 / _s3, the original `main` Watchdog, per-stock
    rows and attacker PnL.

Writes data/watchdog/drip/*.npz, results/drip_test.md and results/drip_test.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from evaluation.cli import guard  # noqa: E402

from evaluation.agents import DripSpoof, Flicker, SameSideTrader  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from watchdog.dataset import DATA_DIR, Source, build_source, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
EPISODES = 50
VERDICT_TAG = "causal_g0_1m"
CONTEXT_TAGS = ["causal_g0_1m_s2", "causal_g0_1m_s3", "main"]
FPR_THRESHOLD = 0.10
SOURCES = (("DRIP-ATK", DripSpoof), ("SAMESIDE", SameSideTrader), ("FLICKER", Flicker))


def main() -> int:
    guard(__doc__, protected=tuple(ROOT / "results" / f for f in ("drip_test.json", "drip_test.md")))
    from sb3_contrib import RecurrentPPO
    from train_spoofer import DECISION_ENV

    models = {}
    for tag in [VERDICT_TAG] + CONTEXT_TAGS:
        path = ROOT / "checkpoints" / "WATCHDOG" / tag / "model.zip"
        if not path.exists():
            print(f"no Watchdog at {path}")
            return 1
        models[tag] = (RecurrentPPO.load(path, device="cpu"), load_obs_norm(path.parent))
    train = load_split("train")
    if not train:
        print("no train split; run watchdog/dataset.py first")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    t0, cache = time.time(), {}
    out_dir = DATA_DIR / "drip"
    out_dir.mkdir(parents=True, exist_ok=True)
    data, pnl = {}, {}
    for tk in TICKERS:
        for name, policy in SOURCES:
            src = build_source(Source(name, tk, policy, DECISION_ENV), "test", EPISODES, cache)
            np.savez_compressed(out_dir / f"{name}_{tk}.npz", **src)
            data[(name, tk)] = src
            if name == "DRIP-ATK":
                pnl[tk] = float(np.mean(src["pnl"]))
        print(f"[{time.time() - t0:5.0f}s] recorded {tk}", flush=True)

    per_model = {}
    for tag, (model, obs_norm) in models.items():
        scored = {k: score_source(model, src, det, obs_norm) for k, src in data.items()}
        per_model[tag] = {
            "pooled": combine(list(scored.values())),
            "drip_only": combine([v for (n, _), v in scored.items() if n == "DRIP-ATK"]),
            "sameside_only": combine([v for (n, _), v in scored.items() if n == "SAMESIDE"]),
            "flicker_only": combine([v for (n, _), v in scored.items() if n == "FLICKER"]),
            "per_stock": {tk: combine([scored[(n, tk)] for n, _ in SOURCES]) for tk in TICKERS},
        }
        print(f"[{time.time() - t0:5.0f}s] scored {tag}", flush=True)

    v = per_model[VERDICT_TAG]
    w_f1, r_f1 = v["pooled"]["watchdog_orders"]["f1"], v["pooled"]["rule_orders"]["f1"]
    ss_fpr = v["sameside_only"]["watchdog_orders"]["fpr"]
    payload = {
        "hypotheses": {"H1": "Watchdog order-level F1 on DRIP-ATK + SAMESIDE + FLICKER >= rule F1",
                       "H2": f"Watchdog order-level FPR on SAMESIDE <= {FPR_THRESHOLD}"},
        "verdict_model": VERDICT_TAG,
        "H1": {"watchdog_f1": w_f1, "rule_f1": r_f1, "verdict": "supported" if w_f1 >= r_f1 else "not supported"},
        "H2": {"watchdog_sameside_fpr": ss_fpr,
               "verdict": "supported" if ss_fpr <= FPR_THRESHOLD else "not supported"},
        "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
        "per_model": per_model, "attacker_pnl_per_episode": pnl, "episodes_per_stock": EPISODES,
        "runtime_s": round(time.time() - t0, 1),
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "drip_test.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (results / "drip_test.md").write_text(render(payload), encoding="utf-8")
    print(f"H1: {payload['H1']['verdict']} (Watchdog F1 {w_f1:.3f}, rule F1 {r_f1:.3f}); "
          f"H2: {payload['H2']['verdict']} (SAMESIDE FPR {ss_fpr:.3f})")
    print(f"wrote {results / 'drip_test.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _row(label: str, b: dict, who: str) -> str:
    m = b[f"{who}_orders"]
    return (f"| {label} | {m['positives']} / {m['n']} | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} | "
            f"{_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | {_f(m['fpr'])} |")


def render(pl: dict) -> str:
    v = pl["per_model"][pl["verdict_model"]]
    L = ["# Pre-registered held-out test: DRIP-ATK", "",
         "Defined and committed before any of its data or results existed; nothing was changed after seeing them.", "",
         f"**H1:** {pl['hypotheses']['H1']}. **Result: {pl['H1']['verdict']}** "
         f"(Watchdog F1 {_f(pl['H1']['watchdog_f1'], 3)}, rule F1 {_f(pl['H1']['rule_f1'], 3)}).", "",
         f"**H2:** {pl['hypotheses']['H2']}. **Result: {pl['H2']['verdict']}** "
         f"(SAMESIDE FPR {_f(pl['H2']['watchdog_sameside_fpr'], 3)}).", "",
         f"Verdict model `checkpoints/WATCHDOG/{pl['verdict_model']}`. Rule tuned on the train split: size ≥ "
         f"{pl['rule']['n_mult']}× depth, cancelled within {pl['rule']['m_events']} events. "
         f"{pl['episodes_per_stock']} episodes per stock of each source.", "",
         "| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |", "|---|---|---|---|---|---|---|"]
    for key, label in (("pooled", "All sources"), ("drip_only", "DRIP-ATK only"),
                       ("sameside_only", "SAMESIDE only"), ("flicker_only", "FLICKER only")):
        L.append(_row(f"Watchdog: {label}", v[key], "watchdog"))
        L.append(_row(f"Rule: {label}", v[key], "rule"))
    L += ["", "Per stock, verdict model (context only):", "",
          "| Bucket | Manipulative / orders | TP / FP / FN / TN | P | R | F1 | FPR |", "|---|---|---|---|---|---|---|"]
    for tk, b in v["per_stock"].items():
        L.append(_row(f"Watchdog: {tk}", b, "watchdog"))
    L += ["", "Other models on the same orders (context only, not used for any verdict):", "",
          "| Model | Pooled F1 | DRIP-ATK recall | SAMESIDE FPR | FLICKER FPR |", "|---|---|---|---|---|"]
    for tag, b in pl["per_model"].items():
        L.append(f"| {tag} | {_f(b['pooled']['watchdog_orders']['f1'], 3)} | {_f(b['drip_only']['watchdog_orders']['recall'], 3)} | "
                 f"{_f(b['sameside_only']['watchdog_orders']['fpr'], 3)} | {_f(b['flicker_only']['watchdog_orders']['fpr'], 3)} |")
    L += ["", "DRIP-ATK PnL per episode (context only): " +
          ", ".join(f"{tk} {x:+,.0f}" for tk, x in pl["attacker_pnl_per_episode"].items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
