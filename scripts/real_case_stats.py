"""Measured order-structure statistics of the simulated attackers, for comparison with documented real cases.

Usage:
    python scripts/real_case_stats.py [--episodes 20]

For each attacker (trained Spoofers with sampled actions, SCRIPTED-ATK, LATEBURST-ATK) on MSFT and INTC, runs
fixed seeds and measures, from the simulation itself:
  * large (spoof) order size relative to the genuine trade size (the lot)
  * how large orders end: cancelled by their owner, filled by the market (run over), or still resting at
    episode end
  * resting time of cancelled large orders in wall-clock seconds, from LOBSTER timestamps
  * share of consecutive large orders placed on the opposite side to the previous one
  * trades on the opposite side per manipulative large order

Some quantities are fixed by the environment design rather than chosen by the agents (spoof size = 10x touch
depth, lot = half touch depth, genuine orders are market orders that always fill); the report marks them.

Writes results/real_case_stats.md and results/real_case_stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import LateBurstSpoof, ModelPolicy, ScriptedSpoof  # noqa: E402
from evaluation.rollout import StepTracker  # noqa: E402

SEED = 900_000


def measure(env: LimitOrderBookEnv, policy, episodes: int) -> dict:
    import torch
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    d = env.day
    records, rest_s, rest_s_manip, flips, sides = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 31 * ep)
        policy.reset(rng)
        tracker, t_at, step = StepTracker(env), [], 0
        while True:
            action = policy.act(obs, env, rng)
            tracker.before_step()
            obs, _, term, trunc, info = env.step(action)
            tracker.after_step(obs, action, info, step, trunc)
            t_at.append(env.t)
            step += 1
            if term or trunc:
                break
        ordered = sorted(tracker.done, key=lambda r: r.placed_step)
        sides += [r.side for r in ordered]
        flips += [a.side != b.side for a, b in zip(ordered, ordered[1:])]
        for rec in ordered:
            records.append(rec)
            if rec.removed_by == "cancel":
                secs = float(d.time[t_at[rec.removed_step]] - d.time[t_at[rec.placed_step]])
                rest_s.append(secs)
                if rec.manipulative:
                    rest_s_manip.append(secs)
    n = len(records)
    manip = [r for r in records if r.manipulative]
    share = lambda k: float(np.mean([r.removed_by == k for r in records])) if n else float("nan")  # noqa: E731
    rest_manip = np.asarray(rest_s_manip)
    return {
        "large_orders": n,
        "manipulative_share": len(manip) / n if n else float("nan"),
        "size_over_lot_median": float(np.median([r.size / env.lot for r in records])) if n else float("nan"),
        "cancelled_share": share("cancel"),
        "filled_by_market_share": share("run_over"),
        "resting_at_episode_end_share": share("episode_end"),
        "resting_seconds_median_cancelled": float(np.median(rest_s)) if rest_s else float("nan"),
        "resting_seconds_median_manipulative": float(np.median(rest_manip)) if len(rest_manip) else float("nan"),
        "share_manipulative_resting_over_1s": float(np.mean(rest_manip > 1.0)) if len(rest_manip) else float("nan"),
        "opposite_side_flip_share": float(np.mean(flips)) if flips else float("nan"),
        "opposite_trades_per_manipulative_median": float(np.median([r.opposite_trades for r in manip])) if manip else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    from train_spoofer import AGENTS, DECISION_ENV

    sources = []
    for agent in ("SPOOFER-02", "SPOOFER-04"):
        path = ROOT / "checkpoints" / agent / "main" / "model.zip"
        if path.exists():
            sources.append((agent, AGENTS[agent]["ticker"], lambda p=path: ModelPolicy(p, deterministic=False),
                            AGENTS[agent]["cfg"]))
    for tk in ("MSFT", "INTC"):
        sources.append(("SCRIPTED-ATK", tk, ScriptedSpoof, DECISION_ENV))
        sources.append(("LATEBURST-ATK", tk, LateBurstSpoof, DECISION_ENV))

    cache, rows = {}, []
    for name, tk, make, cfg in sources:
        if tk not in cache:
            d = load_day(tk)
            cache[tk] = (d, reference_stats(d))
        d, s = cache[tk]
        env = LimitOrderBookEnv(EnvConfig(ticker=tk, **cfg), day=d, stats=s)
        row = {"source": name, "ticker": tk, "lot": env.lot, **measure(env, make(), args.episodes)}
        rows.append(row)
        print(f"{name:14s} {tk}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in row.items() if k not in ("source", "ticker")), flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    payload = {"episodes": args.episodes, "seed": SEED, "rows": rows}
    (out / "real_case_stats.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "real_case_stats.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'real_case_stats.md'}")
    return 0


def _f(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def render(pl: dict) -> str:
    L = ["# Simulated attacker order structure", "",
         f"{pl['episodes']} episodes per source, fixed seeds, sampled actions for trained Spoofers. Resting times are "
         "wall-clock seconds from LOBSTER timestamps. Size ratio is fixed by the environment (spoof = 10× touch "
         "depth, lot = ½ touch depth), not chosen by the agents; genuine orders are market orders and always fill.", "",
         "| Source | Stock | Large orders | Manipulative | Size ÷ lot (fixed) | Cancelled | Filled by market | "
         "Resting at episode end | Median resting s (manipulative) | Manipulative resting > 1 s | "
         "Opposite-side flips | Opposite trades per manipulative order |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pl["rows"]:
        L.append(f"| {r['source']} | {r['ticker']} | {r['large_orders']} | {_f(r['manipulative_share'], True)} | "
                 f"{_f(r['size_over_lot_median'])} | {_f(r['cancelled_share'], True)} | {_f(r['filled_by_market_share'], True)} | "
                 f"{_f(r['resting_at_episode_end_share'], True)} | {_f(r['resting_seconds_median_manipulative'])} | "
                 f"{_f(r['share_manipulative_resting_over_1s'], True)} | {_f(r['opposite_side_flip_share'], True)} | "
                 f"{_f(r['opposite_trades_per_manipulative_median'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
