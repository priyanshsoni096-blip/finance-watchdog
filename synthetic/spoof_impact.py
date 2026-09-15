"""Does a visible spoof move the price in the calibrated synthetic market, and by how much?

Usage:
    python synthetic/spoof_impact.py [--trials 300]

Each trial builds two synthetic markets from the same seed and burn-in, so they are identical. A spoof buy is
placed at the best bid of one (sized like the environment's spoof: 10x the current touch depth); the other is
the control. The mid-price difference after 10, 50 and 200 events is the spoof's emergent impact: no impact
formula is involved, only imbalance followers and noise traders reacting to the visible book. Also measured:
how much of the spoof gets filled by the market within 200 events.

For context, the replay environment's calibrated impact for MSFT and INTC at the same horizons
(lambda x 10 x impact ramp, in spreads) is printed alongside.

Uses configs/synthetic_calibration.json (run synthetic/calibrate.py first).
Writes results/synthetic_spoof_impact.md and results/synthetic_spoof_impact.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthetic.market import BUY, SELL  # noqa: E402
from synthetic.simulator import MarketParams, SyntheticMarket  # noqa: E402

HORIZONS = (10, 50, 200)
SPOOF_K = 10.0
BURN_IN = 2_000


def load_params() -> MarketParams:
    path = ROOT / "configs" / "synthetic_calibration.json"
    if not path.exists():
        raise FileNotFoundError("run synthetic/calibrate.py first")
    return MarketParams(**json.loads(path.read_text())["chosen_params"])


def touch_depth(m: SyntheticMarket) -> float:
    b = m.book
    return (b.volume_at(BUY, b.best_bid()) + b.volume_at(SELL, b.best_ask())) / 2.0


def trial(params: MarketParams, seed: int) -> dict:
    control, spoofed = SyntheticMarket(params, seed), SyntheticMarket(params, seed)
    control.run(BURN_IN)
    spoofed.run(BURN_IN)
    start = control.last_mid
    size = max(1, int(round(SPOOF_K * touch_depth(spoofed))))
    spoofed.book.limit(BUY, spoofed.book.best_bid(), size, "spoofer")
    out, filled = {}, 0
    for k in range(1, max(HORIZONS) + 1):
        control.step()
        filled += sum(f.size for f in spoofed.step() if f.maker_owner == "spoofer")
        if k in HORIZONS:
            out[k] = (spoofed.last_mid - start) - (control.last_mid - start)
    return {"impact_ticks": out, "filled_share": filled / size}


def replay_reference() -> dict:
    path = ROOT / "configs" / "calibration.json"
    if not path.exists():
        return {}
    cal = json.loads(path.read_text())["tickers"]
    ref = {}
    for tk in ("MSFT", "INTC"):
        if tk not in cal or "ramp" not in cal[tk]:
            continue
        ramp = {int(w): r for w, r in cal[tk]["ramp"].items()}
        ws = sorted(ramp)
        ref[tk] = {h: float(cal[tk]["lambda"] * SPOOF_K * np.interp(h, [0] + ws, [0.0] + [ramp[w] for w in ws]))
                   for h in HORIZONS}
    return ref


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=300)
    # sensitivity to the imbalance-follower population, which the calibration statistics do not constrain
    p.add_argument("--p-follower", type=float, default=None)
    p.add_argument("--follower-threshold", type=float, default=None)
    p.add_argument("--tag", default="", help="suffix for output files, e.g. pf030")
    args = p.parse_args()

    from dataclasses import asdict, replace
    from synthetic.calibrate import synthetic_stats

    params = load_params()
    overrides = {k: v for k, v in (("p_follower", args.p_follower), ("follower_threshold", args.follower_threshold))
                 if v is not None}
    params = replace(params, **overrides)
    market_stats = synthetic_stats(params, seed=1)
    trials = [trial(params, 1_000 + i) for i in range(args.trials)]
    rows = {}
    for h in HORIZONS:
        x = np.array([t["impact_ticks"][h] for t in trials])
        rows[h] = {"mean_ticks": float(x.mean()), "ci95_ticks": float(1.96 * x.std(ddof=1) / math.sqrt(len(x))),
                   "share_up": float(np.mean(x > 0)), "share_down": float(np.mean(x < 0))}
    filled = np.array([t["filled_share"] for t in trials])
    payload = {"trials": args.trials, "spoof_k": SPOOF_K, "burn_in": BURN_IN,
               "params": asdict(params), "overrides": overrides, "market_stats": market_stats,
               "impact_by_horizon": rows,
               "spoof_filled_within_200": {"median": float(np.median(filled)), "mean": float(filled.mean())},
               "replay_impact_spreads": replay_reference()}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = "synthetic_spoof_impact" + (f"_{args.tag}" if args.tag else "")
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(render(payload), encoding="utf-8")
    print("overrides: " + (json.dumps(overrides) if overrides else "none") + " | market: "
          + ", ".join(f"{k}={market_stats[k]:.4g}" for k in ("mid_change_share", "range_2000_ticks", "touch_depth", "exec_share")))
    for h, r in rows.items():
        print(f"after {h:3d} events: mean {r['mean_ticks']:+.3f} ticks ± {r['ci95_ticks']:.3f}, "
              f"up {r['share_up']:.0%}, down {r['share_down']:.0%}")
    print(f"spoof filled within 200 events: median {np.median(filled):.1%}, mean {filled.mean():.1%}")
    print(f"wrote {out / (stem + '.md')}")
    return 0


def render(pl: dict) -> str:
    L = ["# Spoof price impact in the synthetic market", "",
         f"{pl['trials']} paired trials (spoofed vs identical control after a {pl['burn_in']}-event burn-in). "
         f"Spoof buy at the best bid, size {pl['spoof_k']:g}× touch depth. The synthetic spread is 1 tick, so ticks equal spreads.", "",
         "Parameter overrides: " + (", ".join(f"{k}={v}" for k, v in pl["overrides"].items()) if pl["overrides"] else "none (calibrated)")
         + ". Resulting market: " + ", ".join(f"{k} {pl['market_stats'][k]:.4g}" for k in
                                             ("mid_change_share", "range_2000_ticks", "touch_depth", "exec_share")) + ".", "",
         "| Events after placement | Mean mid difference (ticks) | 95% CI | Trials up | Trials down |",
         "|---|---|---|---|---|"]
    for h, r in pl["impact_by_horizon"].items():
        L.append(f"| {h} | {r['mean_ticks']:+.3f} | ±{r['ci95_ticks']:.3f} | {r['share_up']:.0%} | {r['share_down']:.0%} |")
    f = pl["spoof_filled_within_200"]
    L += ["", f"Spoof filled by the market within 200 events: median {f['median']:.1%}, mean {f['mean']:.1%}."]
    if pl["replay_impact_spreads"]:
        L += ["", "For comparison, the replay environment's calibrated impact of the same spoof (spreads):", "",
              "| Stock | " + " | ".join(f"{h} events" for h in HORIZONS) + " |",
              "|---|" + "---|" * len(HORIZONS)]
        for tk, ref in pl["replay_impact_spreads"].items():
            L.append(f"| {tk} | " + " | ".join(f"{ref[str(h)] if str(h) in ref else ref[h]:+.2f}" for h in HORIZONS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
