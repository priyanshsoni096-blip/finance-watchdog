"""Basic evaluation: a first, honest read on whether the project's plan holds up.

It checks the three claims from the research question as far as they can be checked BEFORE the
Watchdog exists:

  C1  RL learns meaningful manipulation
      Trained Spoofers vs scripted controls on the same stock: episode PnL (95% CI), how often they
      trade against their own resting spoof, and whether profit exists at all.
  C2  Generalisation to unseen attacks
      The rule-based baseline is tuned on training-pool stocks and scored on held-out AMZN
      (SPOOFER-05) and on a scripted attacker. This is the bar the Watchdog must beat later.
  C3  False positives on legitimate activity
      Baseline flags on intent-less large-order-then-cancel behaviour ("flicker"), and its flag
      rate on real, unlabeled LOBSTER order flow.

Usage:
    python evaluation/basic_eval.py                  # 20 episodes per run
    python evaluation/basic_eval.py --episodes 5     # quick look

Writes results/basic_eval.json and results/basic_eval.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import Flicker, Honest, ModelPolicy, ScriptedSpoof  # noqa: E402
from evaluation.baseline_detector import classification_metrics, real_data_flags, tune  # noqa: E402
from evaluation.rollout import record_episode  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOG", "INTC"]
HELD_OUT = "AMZN"


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    half = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(half)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    from train_spoofer import AGENTS

    t0 = time.time()
    cache = {}

    def env_for(ticker, cfg):
        if ticker not in cache:
            d = load_day(ticker)
            cache[ticker] = (d, reference_stats(d))
        d, s = cache[ticker]
        return LimitOrderBookEnv(EnvConfig(ticker=ticker, **cfg), day=d, stats=s)

    runs = []  # (name, group, ticker, policy, cfg, episodes)
    for agent, spec in AGENTS.items():
        path = ROOT / "checkpoints" / agent / "main" / "model.zip"
        if path.exists():
            group = "rl_heldout" if spec["ticker"] == HELD_OUT else "rl_train"
            runs.append((agent, group, spec["ticker"], ModelPolicy(path), spec["cfg"], args.episodes))
        else:
            print(f"[skip] {agent}: no checkpoint at {path.relative_to(ROOT)}")
    for tk in TICKERS:
        held = "heldout" if tk == HELD_OUT else "train"
        runs.append(("HONEST", f"honest_{held}", tk, Honest(), {}, args.episodes))
        runs.append(("FLICKER", f"flicker_{held}", tk, Flicker(), {}, args.episodes))
        runs.append(("SCRIPTED-ATK", f"scripted_{held}", tk, ScriptedSpoof(), {}, args.episodes))
    # Policies trained on earlier env versions, replayed on the current env: shows each exploit is closed.
    for agent, tag, label in [("SPOOFER-04", "v1_no_selfimpact", "v1 policy, free accumulation"),
                              ("SPOOFER-02", "v2_selfimpact", "v2 policy, alternating swing"),
                              ("SPOOFER-04", "v2_selfimpact", "v2 policy, alternating swing")]:
        old = ROOT / "checkpoints" / agent / tag / "model.zip"
        if old.exists():
            runs.append((f"{agent} ({label})", "exploit_replay", AGENTS[agent]["ticker"], ModelPolicy(old),
                         AGENTS[agent]["cfg"], min(args.episodes, 5)))

    results = []
    for i, (name, group, tk, policy, cfg, n_ep) in enumerate(runs):
        env = env_for(tk, cfg)
        rng = np.random.default_rng(10_000 + i)
        eps = [record_episode(env, policy, rng, seed=50_000 + 97 * j, features=False) for j in range(n_ep)]
        orders = [o for e in eps for o in e["orders"]]
        pnl_mean, pnl_ci = ci95([e["pnl"] for e in eps])
        spread_lot = float(np.median(env.stats.ref_spread)) * env.lot
        row = {
            "agent": name, "group": group, "ticker": tk, "episodes": n_ep, "lot": env.lot,
            "pnl_mean": pnl_mean, "pnl_ci95": pnl_ci,
            "pnl_mean_spread_lots": pnl_mean / spread_lot,
            "frac_profitable_episodes": float(np.mean([e["pnl"] > 0 for e in eps])),
            "trades_per_ep": float(np.mean([e["trades"] for e in eps])),
            "trades_against_own_spoof_per_ep": float(np.mean([e["manip_trades"] for e in eps])),
            "orders_per_ep": len(orders) / n_ep,
            "manipulative_order_frac": float(np.mean([o.manipulative for o in orders])) if orders else float("nan"),
            "_orders": orders,
        }
        results.append(row)
        print(f"[{time.time() - t0:6.0f}s] {name:40s} {tk} pnl={pnl_mean:+12.1f} ±{pnl_ci:9.1f} "
              f"trades/ep={row['trades_per_ep']:7.1f} against-spoof/ep={row['trades_against_own_spoof_per_ep']:6.1f} "
              f"orders/ep={row['orders_per_ep']:5.1f}", flush=True)

    def orders_of(*groups, tickers=None):
        return [o for r in results if r["group"] in groups and (tickers is None or r["ticker"] in tickers)
                for o in r["_orders"]]

    # Baseline tuned on training-pool stocks only (RL train pool + scripted + flicker negatives)
    tune_orders = orders_of("rl_train", "scripted_train", "flicker_train")
    det, grid = tune(tune_orders)
    buckets = {
        "Tuning set (train stocks, all agents)": tune_orders,
        "RL training pool (AAPL/MSFT/GOOG/INTC) + flicker negatives": orders_of("rl_train", "flicker_train"),
        "RL held-out SPOOFER-05 (AMZN) + flicker negatives": orders_of("rl_heldout", "flicker_heldout"),
        "Scripted attacker, train stocks + flicker negatives": orders_of("scripted_train", "flicker_train"),
        "Scripted attacker, AMZN + flicker negatives": orders_of("scripted_heldout", "flicker_heldout"),
        "Flicker only (all orders legitimate)": orders_of("flicker_train", "flicker_heldout"),
    }
    det_rows = {k: classification_metrics(v, det) for k, v in buckets.items()}
    real = {tk: real_data_flags(*cache[tk], det) for tk in TICKERS}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "_orders"} for r in results]
    payload = {"episodes": args.episodes, "detector": {"n_mult": det.n_mult, "m_events": det.m_events},
               "tuning_grid": [{"n_mult": n, "m_events": m, **met} for n, m, met in grid],
               "agents": clean, "detector_buckets": det_rows, "real_flow": real,
               "runtime_s": round(time.time() - t0, 1)}
    (out / "basic_eval.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out / "basic_eval.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nwrote {out / 'basic_eval.json'} and {out / 'basic_eval.md'} in {payload['runtime_s']}s")
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render_markdown(pl: dict) -> str:
    L = ["# Basic evaluation", "",
         "This project demonstrates the feasibility of an RL-based surveillance approach in a realistic "
         "limit-order-book simulation, and evaluates the structural similarity of the emergently-learned "
         "manipulation strategy to documented real-world manipulation cases.", "",
         f"Episodes per run: {pl['episodes']}. 95% CI uses a normal approximation (1.96·sd/√n). "
         f"Runtime {pl['runtime_s']} s. The Watchdog is not built yet, so every detection number below is "
         "the **rule-based baseline** — the bar the Watchdog has to beat.", "",
         "## C1 — Does RL learn meaningful manipulation?", "",
         "PnL is also shown in units of (median spread × lot) so stocks are comparable.", "",
         "| Agent | Group | Ticker | PnL mean ± 95% CI ($) | PnL (spread·lots) | Profitable eps | Trades/ep | Trades against own spoof/ep | Spoof orders/ep | Manipulative orders |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in pl["agents"]:
        L.append(f"| {r['agent']} | {r['group']} | {r['ticker']} | {r['pnl_mean']:+,.1f} ± {r['pnl_ci95']:,.1f} | "
                 f"{r['pnl_mean_spread_lots']:+.1f} | {r['frac_profitable_episodes']:.0%} | {r['trades_per_ep']:.1f} | "
                 f"{r['trades_against_own_spoof_per_ep']:.1f} | {r['orders_per_ep']:.1f} | "
                 f"{_f(r['manipulative_order_frac'] * 100, 0)}% |")
    d = pl["detector"]
    L += ["", "## C2 / C3 — Rule-based baseline detector", "",
          f"Tuned on training stocks only: flag if size ≥ **{d['n_mult']}×** trailing touch depth and cancelled "
          f"within **{d['m_events']}** events without executing. An order is *manipulative* if its owner traded "
          "on the opposite side while it rested.", "",
          "| Bucket | Orders | Manipulative | Legitimate | Precision | Recall | F1 | False-positive rate |",
          "|---|---|---|---|---|---|---|---|"]
    for name, m in pl["detector_buckets"].items():
        L.append(f"| {name} | {m['orders']} | {m['positives']} | {m['negatives']} | {_f(m['precision'])} | "
                 f"{_f(m['recall'])} | {_f(m['f1'])} | {_f(m['fpr'])} |")
    L += ["", "## C3 — Baseline on real, unlabeled LOBSTER order flow", "",
          "Real data has no spoofing labels, so this is a flag *rate*, not a false-positive rate.", "",
          "| Ticker | Events | New orders | Large orders (≥N× depth) | Flags | Flags per 10k events |",
          "|---|---|---|---|---|---|"]
    for tk, r in pl["real_flow"].items():
        L.append(f"| {tk} | {r['events']:,} | {r['new_orders']:,} | {r['large_orders']:,} | {r['flags']:,} | "
                 f"{r['flags_per_10k_events']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
