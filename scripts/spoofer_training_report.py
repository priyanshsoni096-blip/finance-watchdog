"""Spoofer training evidence (execution plan, Phase 2; CP-2): training curves and a behaviour summary.

Usage:
    python scripts/spoofer_training_report.py [--episodes 20]

Figures come from the committed checkpoints/<AGENT>/<tag>/progress.csv (one row per 4,000 training steps); nothing is
retrained. Behaviour is measured by rolling out each final Spoofer (sampled actions, as in every evaluation) on fixed
seeds: spoof size, how long large orders rest, how they end, how long before the first opposite trade, and trades
against the agent's own resting order.

Writes:
  results/figures/spoofer_training_pnl.png        PnL per episode over training, one panel per final Spoofer
  results/figures/spoofer_training_behaviour.png  trades against own resting spoof per 1,000 actions over training
  results/figures/spoofer_failed_runs.png         the two exploit runs, next to the final run
  results/spoofer_behaviour.md / .json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import ModelPolicy  # noqa: E402
from evaluation.rollout import StepTracker  # noqa: E402

SEED = 950_000
INK, MUTED, GRID, SERIES, ZERO = "#2b2b29", "#6b6a64", "#e6e5e0", "#2a78d6", "#9a998f"
FAILED = [("SPOOFER-04", "v1_no_selfimpact", "Free accumulation (before self-impact)"),
          ("SPOOFER-04", "v2_selfimpact", "Alternating-side swing (before impact ramp + cap)"),
          ("SPOOFER-04", "main", "Final: genuine spoofing")]


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.axhline(0, color=ZERO, linewidth=0.8)


def _curve(ax, df: pd.DataFrame, col: str) -> None:
    d = df[["timesteps", col]].dropna()
    x = d["timesteps"] / 1e3
    ax.plot(x, d[col], color=SERIES, alpha=0.25, linewidth=1)
    ax.plot(x, d[col].rolling(8, min_periods=1).mean(), color=SERIES, linewidth=2)


def figures(agents: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def _usd(v, _):
        sign = "−" if v < 0 else ""
        return f"{sign}${abs(v) / 1e3:,.0f}k" if abs(v) >= 1e4 else f"{sign}${abs(v):,.0f}"

    dollars = FuncFormatter(_usd)
    names = []
    for col, fname, ylabel, fmt in (("ep_pnl_mean", "spoofer_training_pnl.png", "PnL per episode", dollars),
                                    ("manip_trades_per_1k", "spoofer_training_behaviour.png",
                                     "Trades against own spoof\nper 1,000 actions", None)):
        fig, axes = plt.subplots(1, len(agents), figsize=(2.6 * len(agents), 2.6), sharex=True)
        for ax, (agent, spec) in zip(axes, agents.items()):
            df = pd.read_csv(ROOT / "checkpoints" / agent / "main" / "progress.csv")
            _curve(ax, df, col)
            _style(ax, f"{agent} · {spec['ticker']}")
            if fmt:
                ax.yaxis.set_major_formatter(fmt)
            ax.set_xlabel("Training steps (thousands)", fontsize=8, color=MUTED)
        fig.supylabel(ylabel, fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200, facecolor="white")
        plt.close(fig)
        names.append(fname)

    fig, axes = plt.subplots(1, len(FAILED), figsize=(3.6 * len(FAILED), 2.8))
    for ax, (agent, tag, label) in zip(axes, FAILED):
        df = pd.read_csv(ROOT / "checkpoints" / agent / tag / "progress.csv")
        _curve(ax, df, "ep_pnl_mean")
        _style(ax, f"{label}\n{agent}/{tag}")
        ax.yaxis.set_major_formatter(dollars)
        ax.set_xlabel("Training steps (thousands)", fontsize=8, color=MUTED)
    fig.supylabel("PnL per episode (own scale per panel)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "spoofer_failed_runs.png", dpi=200, facecolor="white")
    plt.close(fig)
    names.append("spoofer_failed_runs.png")
    return names


def behaviour(agent: str, spec: dict, episodes: int) -> dict:
    import torch
    d = load_day(spec["ticker"])
    env = LimitOrderBookEnv(EnvConfig(ticker=spec["ticker"], **spec["cfg"]), day=d, stats=reference_stats(d))
    policy = ModelPolicy(ROOT / "checkpoints" / agent / "main" / "model.zip", deterministic=False)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    per_step = env.cfg.events_per_step
    orders, waits, pnl, manip = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 31 * ep)
        policy.reset(rng)
        tracker, trades_at, step = StepTracker(env), [], 0
        while True:
            a = policy.act(obs, env, rng)
            tracker.before_step()
            pre = {id(s): s.side for s in env.spoofs}
            obs, _, term, trunc, info = env.step(a)
            tracker.after_step(obs, a, info, step, trunc)
            if info["trade_price"] is not None:
                trades_at.append((step, 1 if a == 1 else -1, set(pre.values())))
            step += 1
            if term or trunc:
                break
        for o in tracker.done:
            orders.append(o)
            if o.manipulative:
                first = next((s for s, side, sides in trades_at
                              if o.placed_step < s <= o.removed_step and side == -o.side and o.side in sides), None)
                if first is not None:
                    waits.append((first - o.placed_step) * per_step)
        pnl.append(info["pnl"])
        manip.append(tracker.manip_trades)
    life = np.array([o.lifetime_events for o in orders if o.lifetime_events is not None], dtype=float)
    share = lambda k: float(np.mean([o.removed_by == k for o in orders])) if orders else float("nan")  # noqa: E731
    return {
        "ticker": spec["ticker"], "episodes": episodes, "pnl_mean": float(np.mean(pnl)),
        "large_orders_per_episode": len(orders) / episodes,
        "manipulative_share": float(np.mean([o.manipulative for o in orders])) if orders else float("nan"),
        "spoof_size_depth_multiple_median": float(np.median([o.depth_mult for o in orders])) if orders else float("nan"),
        "rest_events_median": float(np.median(life)) if life.size else float("nan"),
        "rest_events_p90": float(np.percentile(life, 90)) if life.size else float("nan"),
        "cancelled_share": share("cancel"), "run_over_share": share("run_over"), "episode_end_share": share("episode_end"),
        "wait_to_first_opposite_trade_events_median": float(np.median(waits)) if waits else float("nan"),
        "trades_against_own_spoof_per_episode": float(np.mean(manip)),
    }


def late_training(agent: str) -> dict:
    df = pd.read_csv(ROOT / "checkpoints" / agent / "main" / "progress.csv")
    tail = df.iloc[int(len(df) * 0.9):]
    cols = ["ep_pnl_mean", "frac_buy", "frac_sell", "frac_spoof_buy", "frac_spoof_sell", "frac_cancel",
            "manip_trades_per_1k", "run_overs_per_1k"]
    return {c: float(tail[c].mean()) for c in cols}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    from train_spoofer import AGENTS

    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    names = figures(AGENTS, out)
    rows = {}
    for agent, spec in AGENTS.items():
        rows[agent] = {**behaviour(agent, spec, args.episodes), "late_training": late_training(agent)}
        print(f"{agent}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in rows[agent].items() if k != "late_training"), flush=True)
    payload = {"episodes": args.episodes, "seed": SEED, "figures": names, "agents": rows}
    (ROOT / "results" / "spoofer_behaviour.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (ROOT / "results" / "spoofer_behaviour.md").write_text(render(payload), encoding="utf-8")
    print("wrote results/spoofer_behaviour.md and results/figures/" + ", ".join(names))
    return 0


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def render(pl: dict) -> str:
    L = ["# Spoofer training and behaviour", "",
         "Training curves (from the committed `progress.csv`, one row per 4,000 steps; light line raw, dark line "
         "8-row rolling mean):", "",
         "![PnL over training](figures/spoofer_training_pnl.png)", "",
         "![Trades against own resting spoof over training](figures/spoofer_training_behaviour.png)", "",
         "Earlier runs that found environment exploits, next to the final INTC run (each panel on "
         "its own scale; the exploit runs reached about $250k per episode):", "",
         "![Failed and final runs](figures/spoofer_failed_runs.png)", "",
         f"## Behaviour of the final Spoofers ({pl['episodes']} episodes each, sampled actions, fixed seeds)", "",
         "Market events are the unit of time (10 per agent decision). Spoof size is fixed by design at 10× trailing "
         "touch depth at placement; the median below is its realised multiple.", "",
         "| Spoofer | Stock | PnL / episode | Large orders / ep | Manipulative | Size × depth | Rest (events) median / p90 | "
         "Cancelled / run over / at end | Wait to first opposite trade (events) | Trades against own spoof / ep |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in pl["agents"].items():
        L.append(f"| {a} | {r['ticker']} | {r['pnl_mean']:+,.0f} | {_f(r['large_orders_per_episode'], 1)} | "
                 f"{_f(100 * r['manipulative_share'], 0)}% | {_f(r['spoof_size_depth_multiple_median'], 1)} | "
                 f"{_f(r['rest_events_median'], 0)} / {_f(r['rest_events_p90'], 0)} | "
                 f"{_f(100 * r['cancelled_share'], 0)}% / {_f(100 * r['run_over_share'], 0)}% / {_f(100 * r['episode_end_share'], 0)}% | "
                 f"{_f(r['wait_to_first_opposite_trade_events_median'], 0)} | {_f(r['trades_against_own_spoof_per_episode'], 1)} |")
    L += ["", "## Last 10% of training (rollout means from `progress.csv`)", "",
          "| Spoofer | PnL / episode | Buy | Sell | Spoof buy | Spoof sell | Cancel | Trades against own spoof per 1k | Run-overs per 1k |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, r in pl["agents"].items():
        t = r["late_training"]
        L.append(f"| {a} | {t['ep_pnl_mean']:+,.0f} | {_f(t['frac_buy'], 3)} | {_f(t['frac_sell'], 3)} | "
                 f"{_f(t['frac_spoof_buy'], 3)} | {_f(t['frac_spoof_sell'], 3)} | {_f(t['frac_cancel'], 3)} | "
                 f"{_f(t['manip_trades_per_1k'], 1)} | {_f(t['run_overs_per_1k'], 2)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
