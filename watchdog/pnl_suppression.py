"""PnL suppression: does surveillance acting inside the episode reduce manipulation profit?

Usage:
    python watchdog/pnl_suppression.py [--episodes 50] [--freeze-events 200] [--tag main]

Every source runs the same seeds under three conditions:
  none      no surveillance
  watchdog  the trained Watchdog observes every step; on a flag the exchange cancels the participant's resting
            large orders and rejects new ones for `--freeze-events` market events
  rule      the train-tuned rule, which can only fire when an order is cancelled; same intervention on a flag

Sources are the manipulators that are profitable without surveillance (MSFT and INTC): SPOOFER-02, SPOOFER-04
(sampled actions), SCRIPTED-ATK and LATEBURST-ATK. HONEST and FLICKER on the same stocks measure wrongful
interventions on legitimate activity.

Reported per source and condition: PnL mean with a 95% CI over episodes, suppression % versus no
surveillance, trades against the participant's own resting order, interventions, orders cancelled by
surveillance and blocked spoof attempts.

Limitation: the attackers are frozen and do not adapt to being watched, so suppression is measured against
non-adaptive manipulation.

Writes results/pnl_suppression.md and results/pnl_suppression.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import SPOOF_BUY, SPOOF_SELL, EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import Flicker, Honest, LateBurstSpoof, ModelPolicy, ScriptedSpoof  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import StepTracker  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import OnlineWatchdog, load_obs_norm  # noqa: E402

CONDITIONS = ("none", "watchdog", "rule")
SEED_BASE = 500_000


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(half)


def run_episode(env, policy, rng, seed, condition, watchdog, det, freeze_steps) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset(rng)
    tracker = StepTracker(env)
    if watchdog is not None:
        watchdog.reset()
    freeze_left = triggers = cancelled = blocked = 0
    step = 0
    while True:
        action = policy.act(obs, env, rng)
        if freeze_left > 0 and action in (SPOOF_BUY, SPOOF_SELL):
            action = 0  # NOOP: new large orders are rejected while frozen
            blocked += 1
        tracker.before_step()
        obs, _, term, trunc, info = env.step(action)
        x = tracker.after_step(obs, action, info, step, trunc)
        if condition == "watchdog":
            triggered = watchdog.flag(x)
        elif condition == "rule":
            triggered = any(det.flags(rec) for rec in tracker.last_removed)
        else:
            triggered = False
        if triggered and not (term or trunc):
            triggers += 1
            cancelled += tracker.force_cancel(step)
            freeze_left = freeze_steps
        elif freeze_left > 0:
            freeze_left -= 1
        step += 1
        if term or trunc:
            break
    return {"pnl": float(info["pnl"]), "manip_trades": tracker.manip_trades, "trades": tracker.trades,
            "triggers": triggers, "cancelled": cancelled, "blocked": blocked}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--freeze-events", type=int, default=200)
    p.add_argument("--tag", default="main")
    args = p.parse_args()

    import torch
    from sb3_contrib import RecurrentPPO
    from train_spoofer import AGENTS, DECISION_ENV

    model_path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    if not model_path.exists():
        print(f"no Watchdog at {model_path}")
        return 1
    watchdog = OnlineWatchdog(RecurrentPPO.load(model_path, device="cpu"), load_obs_norm(model_path.parent))
    train = load_split("train")
    if not train:
        print("no train split; run watchdog/dataset.py first")
        return 1
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])
    freeze_steps = -(-args.freeze_events // DECISION_ENV["events_per_step"])

    sources = []
    for agent in ("SPOOFER-02", "SPOOFER-04"):
        spec = AGENTS[agent]
        path = ROOT / "checkpoints" / agent / "main" / "model.zip"
        if path.exists():
            sources.append((agent, spec["ticker"], ModelPolicy(path, deterministic=False), spec["cfg"], "manipulator"))
    for tk in ("MSFT", "INTC"):
        sources.append(("SCRIPTED-ATK", tk, ScriptedSpoof(), DECISION_ENV, "manipulator"))
        sources.append(("LATEBURST-ATK", tk, LateBurstSpoof(), DECISION_ENV, "manipulator"))
        sources.append(("HONEST", tk, Honest(), DECISION_ENV, "legitimate"))
        sources.append(("FLICKER", tk, Flicker(), DECISION_ENV, "legitimate"))

    t0, cache, rows = time.time(), {}, []
    for name, tk, policy, cfg, kind in sources:
        if tk not in cache:
            d = load_day(tk)
            cache[tk] = (d, reference_stats(d))
        d, s = cache[tk]
        env = LimitOrderBookEnv(EnvConfig(ticker=tk, **cfg), day=d, stats=s)
        base = SEED_BASE + zlib.crc32(f"{name}/{tk}".encode()) % 10_000
        row = {"source": name, "ticker": tk, "kind": kind}
        for condition in CONDITIONS:
            rng = np.random.default_rng(base)
            torch.manual_seed(base)
            eps = [run_episode(env, policy, rng, base + 97 * j, condition, watchdog, det, freeze_steps)
                   for j in range(args.episodes)]
            mean, half = ci95([e["pnl"] for e in eps])
            row[condition] = {"pnl_mean": mean, "pnl_ci95": half,
                              **{f"{k}_per_ep": float(np.mean([e[k] for e in eps]))
                                 for k in ("manip_trades", "trades", "triggers", "cancelled", "blocked")}}
        base_pnl = row["none"]["pnl_mean"]
        for condition in ("watchdog", "rule"):
            row[condition]["suppression_pct"] = (100.0 * (base_pnl - row[condition]["pnl_mean"]) / base_pnl
                                                 if base_pnl > 0 else float("nan"))
        rows.append(row)
        print(f"[{time.time() - t0:5.0f}s] {name:14s} {tk} none={base_pnl:+9.1f} "
              f"watchdog={row['watchdog']['pnl_mean']:+9.1f} rule={row['rule']['pnl_mean']:+9.1f}", flush=True)

    payload = {"watchdog": str(model_path.relative_to(ROOT)), "rule": {"n_mult": det.n_mult, "m_events": det.m_events},
               "episodes": args.episodes, "freeze_events": args.freeze_events, "rows": rows,
               "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "pnl_suppression.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "pnl_suppression.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'pnl_suppression.md'}")
    return 0


def _f(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    r = pl["rule"]
    L = ["# PnL suppression", "",
         f"Watchdog `{pl['watchdog']}` and the rule (size ≥ {r['n_mult']}× depth, cancelled within {r['m_events']} events) "
         f"each act inside the episode. On a flag the exchange cancels the participant's resting large orders and "
         f"rejects new ones for {pl['freeze_events']} market events. {pl['episodes']} episodes per source and "
         "condition, identical seeds across conditions. 95% CI is a normal approximation over episodes.", "",
         "Attackers are frozen and do not adapt to surveillance.", "",
         "| Source | Stock | Kind | PnL: none | PnL: Watchdog | Suppression | PnL: rule | Suppression | "
         "Manip. trades/ep none / W / R | Interventions/ep W / R | Blocked spoofs/ep W / R |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in pl["rows"]:
        n, w, rl = row["none"], row["watchdog"], row["rule"]
        L.append(
            f"| {row['source']} | {row['ticker']} | {row['kind']} | {_f(n['pnl_mean'])} ± {_f(n['pnl_ci95'])} | "
            f"{_f(w['pnl_mean'])} ± {_f(w['pnl_ci95'])} | {_f(w['suppression_pct'])}% | "
            f"{_f(rl['pnl_mean'])} ± {_f(rl['pnl_ci95'])} | {_f(rl['suppression_pct'])}% | "
            f"{_f(n['manip_trades_per_ep'])} / {_f(w['manip_trades_per_ep'])} / {_f(rl['manip_trades_per_ep'])} | "
            f"{_f(w['triggers_per_ep'])} / {_f(rl['triggers_per_ep'])} | "
            f"{_f(w['blocked_per_ep'])} / {_f(rl['blocked_per_ep'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
