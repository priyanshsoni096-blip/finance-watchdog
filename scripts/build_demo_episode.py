"""Record annotated demo episodes for the console's demonstration view.

Usage:
    python scripts/build_demo_episode.py [--steps 120]

Replays one episode per scenario in the real replay environment with the committed frozen agents, and records
what actually happened step by step: the top of the order book in dollars, the agent's own resting orders and
trades, the trained Watchdog's flag at that step, and the rule baseline's decision on each order.

Nothing here is simulated for display purposes: prices come from LOBSTER replay, the Watchdog decision is the
committed model's own output, and the manipulation label is the same opposite-side-trade proxy used throughout.

Writes results/demo_episodes.json.
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

from env.lob_env import BUY, CANCEL, NOOP, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import Flicker, LateBurstSpoof, ModelPolicy  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import StepTracker  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import OnlineWatchdog, load_obs_norm  # noqa: E402

ACTIONS = {NOOP: "hold", BUY: "market buy", SELL: "market sell",
           SPOOF_BUY: "place large bid", SPOOF_SELL: "place large ask", CANCEL: "cancel orders"}
LEVELS = 3


def record(scenario: str, kind: str, ticker: str, policy, cfg: dict, seed: int, model, obs_norm, det,
           max_steps: int, cache: dict) -> dict:
    import torch
    if ticker not in cache:
        d = load_day(ticker)
        cache[ticker] = (d, reference_stats(d))
    day, stats = cache[ticker]
    env = LimitOrderBookEnv(EnvConfig(ticker=ticker, **cfg), day=day, stats=stats)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset(rng)
    tracker, wd = StepTracker(env), OnlineWatchdog(model, obs_norm)
    wd.reset()

    steps, step = [], 0
    while True:
        action = policy.act(obs, env, rng)
        t = env.t
        tracker.before_step()
        obs, _, term, trunc, info = env.step(action)
        x = tracker.after_step(obs, action, info, step, trunc)
        flag = wd.flag(x)
        bid_p, bid_s = day.bid_price[t, :LEVELS], day.bid_size[t, :LEVELS]
        ask_p, ask_s = day.ask_price[t, :LEVELS], day.ask_size[t, :LEVELS]
        resting = [{"side": "bid" if s.side > 0 else "ask", "size": float(s.size),
                    "price": round(float(s.price), 4), "age": int(env.t - s.placed_t)} for s in env.spoofs]
        steps.append({
            "step": step,
            "bid": [{"price": round(float(p), 4), "size": float(sz)} for p, sz in zip(bid_p, bid_s)],
            "ask": [{"price": round(float(p), 4), "size": float(sz)} for p, sz in zip(ask_p, ask_s)],
            "mid": round(float(day.mid[t]), 4),
            "action": ACTIONS.get(action, str(action)),
            "trade_price": None if info["trade_price"] is None else round(float(info["trade_price"]), 4),
            "inventory_lots": round(info["inventory"] / env.lot, 1),
            "pnl": round(float(info["pnl"]), 1),
            "resting": resting,
            "watchdog_flag": bool(flag),
            "run_over": bool(info["run_over"]),
        })
        step += 1
        if term or trunc or step >= max_steps:
            break

    # hindsight labels and the rule's verdict, exactly as the evaluation defines them
    manipulative = {r.order_id for r in tracker.done if r.manipulative}
    orders = []
    for r in tracker.done:
        orders.append({
            "id": r.order_id, "side": "bid" if r.side > 0 else "ask", "size": float(r.size),
            "depth_multiple": round(float(r.depth_mult), 1), "placed_step": r.placed_step,
            "removed_step": r.removed_step, "removed_by": r.removed_by,
            "opposite_trades": r.opposite_trades, "manipulative": bool(r.manipulative),
            "rule_flagged": bool(det.flags(r)),
        })
    live_at = {}
    for o in orders:
        end = o["removed_step"] if o["removed_step"] is not None else step
        for s in range(o["placed_step"], min(end, step)):
            live_at.setdefault(s, []).append(o["id"])
    for s in steps:
        ids = live_at.get(s["step"], [])
        s["manipulation_active"] = any(i in manipulative for i in ids)
        s["live_order_ids"] = ids

    flags = sum(s["watchdog_flag"] for s in steps)
    positives = sum(s["manipulation_active"] for s in steps)
    return {
        "scenario": scenario, "kind": kind, "ticker": ticker, "lot": env.lot,
        "events_per_step": env.cfg.events_per_step, "steps": steps, "orders": orders,
        "summary": {
            "steps": len(steps), "manipulation_steps": positives, "watchdog_flagged_steps": flags,
            "large_orders": len(orders),
            "manipulative_orders": sum(o["manipulative"] for o in orders),
            "watchdog_caught": sum(1 for o in orders if o["manipulative"] and any(
                steps[s]["watchdog_flag"] for s in range(o["placed_step"],
                                                         min(o["removed_step"] or len(steps), len(steps))))),
            "rule_flagged_orders": sum(o["rule_flagged"] for o in orders),
            "rule_false_alarms": sum(1 for o in orders if o["rule_flagged"] and not o["manipulative"]),
            "watchdog_false_alarm_orders": sum(
                1 for o in orders if not o["manipulative"] and any(
                    steps[s]["watchdog_flag"] for s in range(o["placed_step"],
                                                             min(o["removed_step"] or len(steps), len(steps))))),
            "final_pnl": steps[-1]["pnl"] if steps else 0.0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    args = p.parse_args()

    from sb3_contrib import RecurrentPPO
    from train_spoofer import AGENTS, DECISION_ENV

    wd_path = ROOT / "checkpoints" / "WATCHDOG" / "main" / "model.zip"
    if not wd_path.exists():
        print("no Watchdog checkpoint")
        return 1
    model = RecurrentPPO.load(wd_path, device="cpu")
    obs_norm = load_obs_norm(wd_path.parent)
    train = load_split("train")
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    spoofer = ROOT / "checkpoints" / "SPOOFER-04" / "main" / "model.zip"
    scenarios = [
        ("RL spoofer (SPOOFER-04)", "manipulator", "INTC",
         lambda: ModelPolicy(spoofer, deterministic=False), AGENTS["SPOOFER-04"]["cfg"], 50_000),
        ("Held-out attacker (LATEBURST-ATK)", "manipulator", "INTC", LateBurstSpoof, DECISION_ENV, 800_017),
        ("Legitimate trader (FLICKER)", "legitimate", "INTC", Flicker, DECISION_ENV, 50_291),
    ]
    cache, out = {}, []
    for name, kind, ticker, make, cfg, seed in scenarios:
        ep = record(name, kind, ticker, make(), cfg, seed, model, obs_norm, det, args.steps, cache)
        s = ep["summary"]
        print(f"{name:36s} steps={s['steps']} manipulation_steps={s['manipulation_steps']} "
              f"flagged_steps={s['watchdog_flagged_steps']} orders={s['large_orders']} "
              f"manipulative={s['manipulative_orders']} rule_flagged={s['rule_flagged_orders']} "
              f"rule_false_alarms={s['rule_false_alarms']}", flush=True)
        out.append(ep)

    payload = {"rule": {"n_mult": det.n_mult, "m_events": det.m_events},
               "watchdog": str(wd_path.relative_to(ROOT)), "episodes": out}
    path = ROOT / "results" / "demo_episodes.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), default=float), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
