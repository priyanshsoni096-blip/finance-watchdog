"""Pre-registered clean held-out test with LATEBURST-ATK (see README, "Pre-registered clean held-out test").

This script, the attacker and the hypothesis were committed before any LATEBURST-ATK data or result existed.
Run it once; do not change the Watchdog, the rule, thresholds or features after seeing its output.

Usage:
    python watchdog/lateburst_test.py

Procedure (fixed in advance):
  * 50 episodes per stock of LATEBURST-ATK and of FLICKER (negatives), both on test-split seeds,
    at the shared decision granularity (10 events per step).
  * The existing Watchdog (checkpoints/WATCHDOG/main, with its saved observation normalisation) and the rule
    tuned on the train split are scored on the same recorded orders.
  * Hypothesis (stock-identity shortcut): Watchdog order-level recall on MSFT+INTC is at least 2x its recall on
    AAPL+GOOG+AMZN. Supported if the ratio is >= 2, not supported if below 2, undetermined if both recalls are 0.

Writes data/watchdog/lateburst/*.npz, results/lateburst_test.md and results/lateburst_test.json.
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

from evaluation.agents import Flicker, LateBurstSpoof  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from watchdog.dataset import DATA_DIR, Source, build_source, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
ONE_TICK = ["MSFT", "INTC"]
WIDE = ["AAPL", "GOOG", "AMZN"]
EPISODES = 50
RATIO_THRESHOLD = 2.0


def verdict(recall_one_tick: float, recall_wide: float) -> tuple[str, float]:
    if recall_one_tick == 0 and recall_wide == 0:
        return "undetermined (recall 0 on both groups)", float("nan")
    if recall_wide == 0:
        return "supported (recall 0 on AAPL+GOOG+AMZN only)", float("inf")
    ratio = recall_one_tick / recall_wide
    return ("supported" if ratio >= RATIO_THRESHOLD else "not supported"), ratio


def main() -> int:
    from sb3_contrib import RecurrentPPO
    from train_spoofer import DECISION_ENV

    model_path = ROOT / "checkpoints" / "WATCHDOG" / "main" / "model.zip"
    if not model_path.exists():
        print(f"no Watchdog at {model_path}")
        return 1
    model = RecurrentPPO.load(model_path, device="cpu")
    obs_norm = load_obs_norm(model_path.parent)
    train = load_split("train")
    if not train:
        print("no train split; run watchdog/dataset.py first")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    t0, cache = time.time(), {}
    out_dir = DATA_DIR / "lateburst"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_attack, scored_flicker, pnl = {}, {}, {}
    for tk in TICKERS:
        attack = build_source(Source("LATEBURST-ATK", tk, LateBurstSpoof, DECISION_ENV), "test", EPISODES, cache)
        flicker = build_source(Source("FLICKER", tk, Flicker, DECISION_ENV), "test", EPISODES, cache)
        np.savez_compressed(out_dir / f"LATEBURST-ATK_{tk}.npz", **attack)
        np.savez_compressed(out_dir / f"FLICKER_{tk}.npz", **flicker)
        scored_attack[tk] = score_source(model, attack, det, obs_norm)
        scored_flicker[tk] = score_source(model, flicker, det, obs_norm)
        pnl[tk] = float(np.mean(attack["pnl"]))
        print(f"[{time.time() - t0:5.0f}s] {tk}: attack orders={len(scored_attack[tk]['order_truth'])} "
              f"manipulative={int(np.sum(scored_attack[tk]['order_truth']))} pnl/ep={pnl[tk]:+.1f}", flush=True)

    def group(tickers):
        return combine([scored_attack[t] for t in tickers] + [scored_flicker[t] for t in tickers])

    per_stock = {tk: group([tk]) for tk in TICKERS}
    one_tick, wide, overall = group(ONE_TICK), group(WIDE), group(TICKERS)
    r1, rw = one_tick["watchdog_orders"]["recall"], wide["watchdog_orders"]["recall"]
    result, ratio = verdict(r1, rw)

    payload = {
        "hypothesis": "Watchdog order-level recall on MSFT+INTC >= 2x recall on AAPL+GOOG+AMZN",
        "watchdog_recall_msft_intc": r1, "watchdog_recall_aapl_goog_amzn": rw, "ratio": ratio, "verdict": result,
        "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
        "overall": overall, "groups": {"MSFT+INTC": one_tick, "AAPL+GOOG+AMZN": wide},
        "per_stock": per_stock, "attacker_pnl_per_episode": pnl, "episodes_per_stock": EPISODES,
        "runtime_s": round(time.time() - t0, 1),
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "lateburst_test.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (results / "lateburst_test.md").write_text(render(payload), encoding="utf-8")
    print(f"verdict: {result} (recall MSFT+INTC {r1:.3f}, AAPL+GOOG+AMZN {rw:.3f}, ratio {ratio:.2f})")
    print(f"wrote {results / 'lateburst_test.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(pl: dict) -> str:
    L = ["# Pre-registered held-out test: LATEBURST-ATK", "",
         "Defined and committed before any of its data or results existed; nothing was changed after seeing them.", "",
         f"**Hypothesis:** {pl['hypothesis']}.", "",
         f"**Result: {pl['verdict']}.** Watchdog order-level recall MSFT+INTC = {_f(pl['watchdog_recall_msft_intc'], 3)}, "
         f"AAPL+GOOG+AMZN = {_f(pl['watchdog_recall_aapl_goog_amzn'], 3)}, ratio = {_f(pl['ratio'])}.", "",
         f"Rule tuned on the train split: size ≥ {pl['rule']['n_mult']}× depth, cancelled within {pl['rule']['m_events']} "
         f"events. {pl['episodes_per_stock']} episodes per stock of LATEBURST-ATK plus FLICKER negatives.", "",
         "| Group | Manipulative / orders | Watchdog P | Watchdog R | Watchdog FPR | Rule P | Rule R | Rule FPR | Median delay W / R |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = [("All stocks", pl["overall"])] + list(pl["groups"].items()) + list(pl["per_stock"].items())
    for name, b in rows:
        w, r = b["watchdog_orders"], b["rule_orders"]
        L.append(f"| {name} | {w['positives']} / {w['n']} | {_f(w['precision'])} | {_f(w['recall'])} | {_f(w['fpr'])} | "
                 f"{_f(r['precision'])} | {_f(r['recall'])} | {_f(r['fpr'])} | "
                 f"{_f(b['watchdog_median_delay'], 0)} / {_f(b['rule_median_delay'], 0)} |")
    L += ["", "Attacker PnL per episode (context only): " +
          ", ".join(f"{tk} {v:+,.0f}" for tk, v in pl["attacker_pnl_per_episode"].items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
