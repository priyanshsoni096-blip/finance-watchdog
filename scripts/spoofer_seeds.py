"""Does the Spoofers' learned profitability depend on the training seed?

Usage:
    python scripts/spoofer_seeds.py [--episodes 20] [--tags main seed11 seed23]

The headline Spoofer results came from one training seed each. This runs every available seed of the two
profitable Spoofers (SPOOFER-02 on MSFT, SPOOFER-04 on INTC) through the same evaluation and reports:

  * PnL per episode, mean with a 95% CI over episodes (actions sampled from the policy, as everywhere else)
  * the same policy with impact_lambda=0, so profit that does not come from the spoof is visible
  * trades made against the agent's own resting order, and the share of its large orders that are manipulative
  * across seeds: the mean of the seed means with a Student-t 95% CI

A seed that converged to not trading shows up as ~0 PnL with no manipulative orders; that is a real outcome of
the training process, not an error, and it is reported rather than dropped.

Writes results/spoofer_seeds.md and results/spoofer_seeds.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import ModelPolicy  # noqa: E402
from evaluation.rollout import record_episode  # noqa: E402
from watchdog.multiseed_summary import mean_ci95  # noqa: E402

AGENTS_TO_CHECK = ("SPOOFER-02", "SPOOFER-04")
SEED_BASE = 950_000


def ci95_episodes(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(half)


def evaluate(agent: str, tag: str, cfg: dict, ticker: str, episodes: int, cache: dict) -> dict:
    import torch
    path = ROOT / "checkpoints" / agent / tag / "model.zip"
    if ticker not in cache:
        d = load_day(ticker)
        cache[ticker] = (d, reference_stats(d))
    d, s = cache[ticker]
    out = {"agent": agent, "tag": tag, "ticker": ticker}
    for label, extra in (("calibrated", {}), ("no_impact", {"impact_lambda": 0.0})):
        env = LimitOrderBookEnv(EnvConfig(ticker=ticker, **cfg, **extra), day=d, stats=s)
        policy = ModelPolicy(path, deterministic=False)
        base = SEED_BASE + zlib.crc32(f"{agent}/{tag}".encode()) % 10_000
        rng = np.random.default_rng(base)
        torch.manual_seed(base)
        eps = [record_episode(env, policy, rng, base + 97 * j, features=False) for j in range(episodes)]
        orders = [o for e in eps for o in e["orders"]]
        mean, half = ci95_episodes([e["pnl"] for e in eps])
        out[label] = {
            "pnl_mean": mean, "pnl_ci95": half,
            "trades_per_ep": float(np.mean([e["trades"] for e in eps])),
            "trades_against_own_spoof_per_ep": float(np.mean([e["manip_trades"] for e in eps])),
            "manipulative_order_share": float(np.mean([o.manipulative for o in orders])) if orders else float("nan"),
            "profitable_episode_share": float(np.mean([e["pnl"] > 0 for e in eps])),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--tags", nargs="*", default=["main", "seed11", "seed23"])
    args = p.parse_args()

    from train_spoofer import AGENTS

    cache, rows = {}, []
    for agent in AGENTS_TO_CHECK:
        spec = AGENTS[agent]
        for tag in args.tags:
            if not (ROOT / "checkpoints" / agent / tag / "model.zip").exists():
                print(f"[skip] {agent}/{tag}: no checkpoint")
                continue
            row = evaluate(agent, tag, spec["cfg"], spec["ticker"], args.episodes, cache)
            rows.append(row)
            c, n = row["calibrated"], row["no_impact"]
            print(f"{agent} {tag:7s} {row['ticker']}: PnL {c['pnl_mean']:+10.1f} ± {c['pnl_ci95']:8.1f} | "
                  f"no-impact {n['pnl_mean']:+10.1f} | against-spoof/ep {c['trades_against_own_spoof_per_ep']:6.1f} | "
                  f"manipulative {c['manipulative_order_share']:.2f}", flush=True)

    across = {}
    for agent in AGENTS_TO_CHECK:
        seeds = [r for r in rows if r["agent"] == agent]
        if not seeds:
            continue
        mean, half = mean_ci95([r["calibrated"]["pnl_mean"] for r in seeds])
        nmean, nhalf = mean_ci95([r["no_impact"]["pnl_mean"] for r in seeds])
        across[agent] = {
            "ticker": seeds[0]["ticker"], "seeds": len(seeds),
            "pnl_mean": mean, "pnl_ci95": half, "no_impact_mean": nmean, "no_impact_ci95": nhalf,
            "seeds_profitable": int(sum(r["calibrated"]["pnl_mean"] > 0 for r in seeds)),
            "seeds_that_traded": int(sum(r["calibrated"]["trades_per_ep"] > 1 for r in seeds)),
        }

    payload = {"episodes": args.episodes, "tags": args.tags, "per_seed": rows, "across_seeds": across}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "spoofer_seeds.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "spoofer_seeds.md").write_text(render(payload), encoding="utf-8")
    for agent, a in across.items():
        print(f"{agent} across {a['seeds']} seeds: PnL {a['pnl_mean']:+,.1f} ± {a['pnl_ci95']:,.1f} "
              f"(no impact {a['no_impact_mean']:+,.1f}), traded in {a['seeds_that_traded']}/{a['seeds']} seeds")
    print(f"wrote {out / 'spoofer_seeds.md'}")
    return 0


def _f(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    L = ["# Spoofer results across training seeds", "",
         f"{pl['episodes']} episodes per seed, actions sampled from the policy. The no-impact column is the same "
         "policy with the price-impact model switched off: profit that survives there does not come from spoofing. "
         "Across-seed CIs are Student-t over the seed means.", "",
         "| Agent | Stock | Seed | PnL per episode | No impact | Trades/ep | Against own spoof/ep | Manipulative orders | Profitable episodes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in pl["per_seed"]:
        c, n = r["calibrated"], r["no_impact"]
        share = c["manipulative_order_share"]
        L.append(f"| {r['agent']} | {r['ticker']} | {r['tag']} | {_f(c['pnl_mean'])} ± {_f(c['pnl_ci95'])} | "
                 f"{_f(n['pnl_mean'])} | {_f(c['trades_per_ep'])} | {_f(c['trades_against_own_spoof_per_ep'])} | "
                 f"{'n/a' if not np.isfinite(share) else f'{share:.0%}'} | {c['profitable_episode_share']:.0%} |")
    L += ["", "| Agent | Stock | Seeds | PnL across seeds | No impact across seeds | Seeds that traded | Seeds profitable |",
          "|---|---|---|---|---|---|---|"]
    for agent, a in pl["across_seeds"].items():
        L.append(f"| {agent} | {a['ticker']} | {a['seeds']} | {_f(a['pnl_mean'])} ± {_f(a['pnl_ci95'])} | "
                 f"{_f(a['no_impact_mean'])} ± {_f(a['no_impact_ci95'])} | {a['seeds_that_traded']}/{a['seeds']} | "
                 f"{a['seeds_profitable']}/{a['seeds']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
