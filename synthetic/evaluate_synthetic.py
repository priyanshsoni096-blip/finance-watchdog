"""Sim-to-sim robustness: frozen agents and the Watchdog inside the calibrated synthetic market.

Usage:
    python synthetic/evaluate_synthetic.py [--episodes 30]

Nothing is trained here. Each source runs the same seeds under two conditions:
  reactive  imbalance followers see the agent's resting orders (spoofs can move the price)
  blind     counterfactual: followers ignore the agent's own resting volume
Manipulation gain = reactive PnL - blind PnL, measured as a paired difference over episodes. It is what spoofing
earns through the market's reaction alone, with no impact formula.

Detection uses the reactive episodes: the main Watchdog (with its saved observation normalisation) and the rule
tuned on the replay train split, both scored on the same recorded orders exactly as in
watchdog/evaluate_watchdog.py.

Writes results/synthetic_eval.md and results/synthetic_eval.json.
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

from env.lob_env import EnvConfig  # noqa: E402
from evaluation.agents import Flicker, Honest, LateBurstSpoof, ModelPolicy, ScriptedSpoof  # noqa: E402
from evaluation.baseline_detector import tune  # noqa: E402
from evaluation.rollout import record_episode  # noqa: E402
from synthetic.spoof_impact import load_params  # noqa: E402
from synthetic.synthetic_env import SyntheticSpoofEnv  # noqa: E402
from watchdog.dataset import _orders_to_array, load_split, orders_from_array  # noqa: E402
from watchdog.evaluate_watchdog import combine, score_source  # noqa: E402
from watchdog.watchdog_env import load_obs_norm  # noqa: E402

SEED_BASE = 700_000


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), (float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--warmup-events", type=int, default=6_000)
    args = p.parse_args()

    import torch
    from sb3_contrib import RecurrentPPO
    from train_spoofer import AGENTS, DECISION_ENV

    params = load_params()
    wd_path = ROOT / "checkpoints" / "WATCHDOG" / "main" / "model.zip"
    model = RecurrentPPO.load(wd_path, device="cpu")
    obs_norm = load_obs_norm(wd_path.parent)
    train = load_split("train")
    det, _ = tune([o for s in train.values() for _, o in orders_from_array(s["orders"])])

    sources = []
    for agent in ("SPOOFER-02", "SPOOFER-04"):
        path = ROOT / "checkpoints" / agent / "main" / "model.zip"
        if path.exists():
            sources.append((agent, "manipulator", lambda p=path: ModelPolicy(p, deterministic=False), AGENTS[agent]["cfg"]))
    sources += [("SCRIPTED-ATK", "manipulator", ScriptedSpoof, DECISION_ENV),
                ("LATEBURST-ATK", "manipulator", LateBurstSpoof, DECISION_ENV),
                ("HONEST", "legitimate", Honest, DECISION_ENV),
                ("FLICKER", "legitimate", Flicker, DECISION_ENV)]

    t0, rows, scored = time.time(), [], {}
    for name, kind, make, cfg in sources:
        policy = make()
        base = SEED_BASE + zlib.crc32(name.encode()) % 10_000
        row = {"source": name, "kind": kind}
        pnl_by = {}
        for condition in ("reactive", "blind"):
            env = SyntheticSpoofEnv(EnvConfig(**cfg), params, warmup_events=args.warmup_events,
                                    blind_followers=(condition == "blind"))
            rng = np.random.default_rng(base)
            torch.manual_seed(base)
            eps = [record_episode(env, policy, rng, base + 97 * j, features=(condition == "reactive"))
                   for j in range(args.episodes)]
            orders = [o for e in eps for o in e["orders"]]
            pnl_by[condition] = [e["pnl"] for e in eps]
            mean, half = ci95(pnl_by[condition])
            row[condition] = {
                "pnl_mean": mean, "pnl_ci95": half,
                "trades_per_ep": float(np.mean([e["trades"] for e in eps])),
                "trades_against_own_spoof_per_ep": float(np.mean([e["manip_trades"] for e in eps])),
                "large_orders_per_ep": len(orders) / len(eps),
                "manipulative_share": float(np.mean([o.manipulative for o in orders])) if orders else float("nan"),
                "filled_by_market_share": float(np.mean([o.removed_by == "run_over" for o in orders])) if orders else float("nan"),
            }
            if condition == "reactive":
                lengths = np.array([len(e["y"]) for e in eps])
                src = {"X": np.concatenate([e["X"] for e in eps]), "y": np.concatenate([e["y"] for e in eps]),
                       "ep_offsets": np.concatenate([[0], np.cumsum(lengths)]),
                       "orders": _orders_to_array([e["orders"] for e in eps])}
                scored[name] = score_source(model, src, det, obs_norm)
        gain_mean, gain_half = ci95(np.array(pnl_by["reactive"]) - np.array(pnl_by["blind"]))
        row["manipulation_gain_mean"], row["manipulation_gain_ci95"] = gain_mean, gain_half
        s = scored[name]
        row["watchdog_step_flag_rate"] = float(np.mean(s["step_flags"]))
        row["positive_step_rate"] = float(np.mean(s["step_labels"]))
        rows.append(row)
        print(f"[{time.time() - t0:5.0f}s] {name:14s} reactive={row['reactive']['pnl_mean']:+9.1f} "
              f"blind={row['blind']['pnl_mean']:+9.1f} gain={gain_mean:+8.1f}±{gain_half:.1f} "
              f"against-spoof/ep={row['reactive']['trades_against_own_spoof_per_ep']:.1f}", flush=True)

    manip = [scored[r["source"]] for r in rows if r["kind"] == "manipulator"]
    legit = [scored[r["source"]] for r in rows if r["kind"] == "legitimate"]
    detection = {
        "Manipulators + legitimate": combine(manip + legit),
        "Legitimate only": combine(legit),
        **{f"{r['source']} + legitimate": combine([scored[r["source"]]] + legit)
           for r in rows if r["kind"] == "manipulator"},
    }
    payload = {"episodes": args.episodes, "warmup_events": args.warmup_events, "market_params": params.__dict__,
               "rule": {"n_mult": det.n_mult, "m_events": det.m_events}, "watchdog": str(wd_path.relative_to(ROOT)),
               "sources": rows, "detection": detection, "runtime_s": round(time.time() - t0, 1)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "synthetic_eval.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "synthetic_eval.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {out / 'synthetic_eval.md'}")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    r = pl["rule"]
    L = ["# Sim-to-sim robustness: synthetic agent-based market", "",
         "Frozen agents (nothing trained here) in the calibrated synthetic market, where prices come from order "
         f"matching. {pl['episodes']} episodes per source and condition; identical seeds across conditions. "
         "Manipulation gain = reactive PnL − blind-follower PnL (paired, 95% CI).", "",
         "## Does manipulation still pay?", "",
         "| Source | Kind | PnL reactive | PnL blind followers | Manipulation gain | Trades against own spoof/ep | Large orders/ep | Manipulative | Filled by market |",
         "|---|---|---|---|---|---|---|---|---|"]
    for row in pl["sources"]:
        re_, bl = row["reactive"], row["blind"]
        L.append(f"| {row['source']} | {row['kind']} | {_f(re_['pnl_mean'], 1)} ± {_f(re_['pnl_ci95'], 1)} | "
                 f"{_f(bl['pnl_mean'], 1)} ± {_f(bl['pnl_ci95'], 1)} | {_f(row['manipulation_gain_mean'], 1)} ± "
                 f"{_f(row['manipulation_gain_ci95'], 1)} | {_f(re_['trades_against_own_spoof_per_ep'], 1)} | "
                 f"{_f(re_['large_orders_per_ep'], 1)} | {_f(re_['manipulative_share'] * 100 if np.isfinite(re_['manipulative_share']) else float('nan'), 0)}% | "
                 f"{_f(re_['filled_by_market_share'] * 100 if np.isfinite(re_['filled_by_market_share']) else float('nan'), 0)}% |")
    L += ["", "## Detection in the synthetic market", "",
          f"Watchdog `{pl['watchdog']}` (trained on replay data only) vs the rule tuned on the replay train split "
          f"(size ≥ {r['n_mult']}× depth, cancelled within {r['m_events']} events), on the same recorded orders.", "",
          "| Bucket | Manipulative / orders | Watchdog P | Watchdog R | Watchdog F1 | Watchdog FPR | Rule P | Rule R | Rule F1 | Rule FPR |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in pl["detection"].items():
        w, rl = b["watchdog_orders"], b["rule_orders"]
        L.append(f"| {name} | {w['positives']} / {w['n']} | {_f(w['precision'])} | {_f(w['recall'])} | {_f(w['f1'])} | "
                 f"{_f(w['fpr'])} | {_f(rl['precision'])} | {_f(rl['recall'])} | {_f(rl['f1'])} | {_f(rl['fpr'])} |")
    L += ["", "| Source | Positive step rate | Watchdog step flag rate |", "|---|---|---|"]
    for row in pl["sources"]:
        L.append(f"| {row['source']} | {row['positive_step_rate']:.3f} | {row['watchdog_step_flag_rate']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
