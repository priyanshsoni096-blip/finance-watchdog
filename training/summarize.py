"""Summarize Spoofer training progress: is genuine spoof-and-cancel emerging, or a degenerate policy?

Usage: python training/summarize.py SPOOFER-01 [--tag main] [--buckets 5]

Splits progress.csv into equal time buckets and prints the mean of each behaviour column, so
trends (reward rising, trades against a resting spoof rising, run-overs falling) are visible
without plotting. Episode columns are averaged only over rollouts that finished an episode.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLS = ["ep_reward_mean", "ep_pnl_mean", "frac_noop", "frac_buy", "frac_sell", "frac_spoof_buy",
        "frac_spoof_sell", "frac_cancel", "manip_trades_per_1k", "run_overs_per_1k"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agent")
    p.add_argument("--tag", default="main")
    p.add_argument("--buckets", type=int, default=5)
    args = p.parse_args()

    path = ROOT / "checkpoints" / args.agent / args.tag / "progress.csv"
    if not path.exists():
        print(f"no progress file at {path}")
        return 1
    df = pd.read_csv(path)
    if df.empty:
        print("progress file is empty")
        return 1
    n = min(args.buckets, len(df))
    df["bucket"] = np.minimum((np.arange(len(df)) * n) // len(df), n - 1)
    g = df.groupby("bucket")
    table = g[COLS].mean()
    table.insert(0, "steps_to", g["timesteps"].max())
    table.insert(1, "episodes", g["episodes"].sum())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"{args.agent}/{args.tag}: {len(df)} rollouts, {int(df['timesteps'].iloc[-1])} steps, "
          f"{df['wall_s'].iloc[-1] / 60:.1f} min")
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
