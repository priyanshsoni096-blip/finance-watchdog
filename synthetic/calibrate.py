"""Calibrate the synthetic market to real market statistics (not to spoof profitability).

Usage:
    python synthetic/calibrate.py

Targets are measured from the real MSFT and INTC LOBSTER data (after the 30k-event warmup) and averaged:
median spread (ticks), share of events where the mid changes, median mid range over 2,000 events (ticks),
median touch depth (shares), and the event mix (share of new orders, cancellations and executions).
A small grid over the background-agent parameters is scored only on distance to those targets
(sum of squared log ratios). Spoof price impact is measured separately and is not part of the objective.

MSFT and INTC are used rather than all four training stocks: pooling 1-tick stocks with AAPL/GOOG (15-28 tick
spreads) gives statistics that describe no real market, and the 1-tick stocks are where spoofing is profitable.

Writes configs/synthetic_calibration.json.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.lobster_data import DELETE, EXEC_HIDDEN, EXEC_VISIBLE, NEW, PARTIAL_CANCEL, load_day  # noqa: E402
from synthetic.market import BUY, SELL  # noqa: E402
from synthetic.simulator import MarketParams, SyntheticMarket  # noqa: E402

TARGET_TICKERS = ("MSFT", "INTC")
KEYS = ("spread_ticks", "mid_change_share", "range_2000_ticks", "touch_depth", "cancel_share", "exec_share")
GRID = {
    "cancel_hazard": (0.002, 0.0035, 0.005),
    "noise_p_market": (0.04, 0.07),
    "noise_depth_geom_p": (0.25, 0.35, 0.5),
    "p_follower": (0.05, 0.15),
}
SEEDS = (1, 2)
OUT = ROOT / "configs" / "synthetic_calibration.json"


def real_targets() -> dict:
    per = []
    for tk in TARGET_TICKERS:
        d = load_day(tk)
        sl = slice(30_000, None)
        et, mid = d.event_type[sl], d.mid[sl]
        dm = np.abs(np.diff(mid))
        ok = np.isfinite(dm)
        ranges = [np.nanmax(mid[i:i + 2000]) - np.nanmin(mid[i:i + 2000]) for i in range(0, len(mid) - 2000, 2000)]
        per.append({
            "spread_ticks": float(np.nanmedian(d.spread[sl]) / 0.01),
            "mid_change_share": float(np.mean(dm[ok] > 0)),
            "range_2000_ticks": float(np.median(ranges) / 0.01),
            "touch_depth": float(np.median((d.ask_size[sl, 0] + d.bid_size[sl, 0]) / 2)),
            "cancel_share": float(np.mean(np.isin(et, [PARTIAL_CANCEL, DELETE]))),
            "exec_share": float(np.mean(np.isin(et, [EXEC_VISIBLE, EXEC_HIDDEN]))),
            "new_share": float(np.mean(et == NEW)),
        })
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def synthetic_stats(params: MarketParams, seed: int, n: int = 60_000, burn: int = 10_000) -> dict:
    m = SyntheticMarket(params, seed)
    m.run(burn)
    m.counts = {k: 0 for k in m.counts}
    mids, spreads, depths = np.empty(n), np.empty(n), []
    for i in range(n):
        m.step()
        mids[i] = m.last_mid
        spreads[i] = m.book.best_ask() - m.book.best_bid()
        if i % 10 == 0:
            depths.append((m.book.volume_at(BUY, m.book.best_bid()) + m.book.volume_at(SELL, m.book.best_ask())) / 2)
    c = m.counts
    # LOBSTER logs one execution message per resting order hit, so fills count as execution events
    events = c["limit"] + c["cancel"] + c["fills"]
    ranges = [np.ptp(mids[i:i + 2000]) for i in range(0, n - 2000, 2000)]
    return {
        "spread_ticks": float(np.median(spreads)),
        "mid_change_share": float(np.mean(np.diff(mids) != 0)),
        "range_2000_ticks": float(np.median(ranges)),
        "touch_depth": float(np.median(depths)),
        "cancel_share": c["cancel"] / events,
        "exec_share": c["fills"] / events,
        "new_share": c["limit"] / events,
    }


def score(stats: dict, targets: dict) -> float:
    eps = 1e-6
    return float(sum(np.log((stats[k] + eps) / (targets[k] + eps)) ** 2 for k in KEYS))


def main() -> int:
    t0 = time.time()
    targets = real_targets()
    print("targets (mean of MSFT, INTC): " + ", ".join(f"{k}={v:.4g}" for k, v in targets.items()), flush=True)
    results = []
    names = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = replace(MarketParams(), **dict(zip(names, values)))
        per_seed = [synthetic_stats(params, s) for s in SEEDS]
        stats = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
        results.append((score(stats, targets), dict(zip(names, values)), stats))
        print(f"[{time.time() - t0:5.0f}s] {dict(zip(names, values))} score={results[-1][0]:.3f} "
              + " ".join(f"{k}={stats[k]:.4g}" for k in KEYS), flush=True)
    results.sort(key=lambda r: r[0])
    best_score, best_values, best_stats = results[0]
    chosen = asdict(replace(MarketParams(), **best_values))
    payload = {
        "target_tickers": list(TARGET_TICKERS), "targets": targets, "objective_keys": list(KEYS),
        "chosen_params": chosen, "chosen_stats": best_stats, "chosen_score": best_score,
        "top5": [{"score": s, "params": v, "stats": st} for s, v, st in results[:5]],
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nchosen: " + json.dumps(best_values))
    print(f"{'statistic':18s} {'real target':>12s} {'synthetic':>12s}")
    for k in KEYS + ("new_share",):
        print(f"{k:18s} {targets[k]:12.4g} {best_stats[k]:12.4g}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
