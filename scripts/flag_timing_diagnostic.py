"""Diagnostic for CP-7: at which point in a manipulative order's life does the Watchdog flag?

Usage:
    python scripts/flag_timing_diagnostic.py [--tag main] [--split test]

Only the train and test splits are allowed: this informs a design decision, so held-out and LATEBURST data stay
unseen (HANDOFF §9). For every manipulative order the steps it rests are split by phase:

  before    no opposite-side trade by its owner yet (the label is set with hindsight, so the evidence that makes
            the order manipulative does not exist yet at these steps)
  trade     a step on which the owner trades on the opposite side
  after     at least one opposite trade already happened, no trade on this step

and the Watchdog's flag rate is reported per phase and per source, plus order-level recall by the number of
opposite trades and by the wait (steps) before the first one. The trade-direction feature is index 44 and the
signed resting-side imbalance index 45 (evaluation/rollout.py).

Prints to the terminal.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.rollout import BOOK_DIM  # noqa: E402
from watchdog.dataset import load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import flag_episode, load_obs_norm  # noqa: E402

TRADE = BOOK_DIM + 4


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--split", default="test", choices=["train", "test"])
    args = p.parse_args()

    from sb3_contrib import RecurrentPPO
    path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    model = RecurrentPPO.load(path, device="cpu")
    obs_norm = load_obs_norm(path.parent)
    data = load_split(args.split)

    phase_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # source -> phase -> [flags, steps]
    by_trades = defaultdict(lambda: [0, 0])                          # opposite trades -> [caught, orders]
    by_wait = defaultdict(lambda: [0, 0])                            # wait bucket -> [caught, orders]
    first_flag_vs_trade = []                                         # first flag step - first trade step
    for name, src in sorted(data.items()):
        if not name.startswith("SPOOFER"):
            continue
        off = src["ep_offsets"]
        flags = [flag_episode(model, src["X"][a:b], obs_norm) for a, b in zip(off[:-1], off[1:])]
        for ep, o in orders_from_array(src["orders"]):
            if not o.manipulative:
                continue
            X = src["X"][off[ep]:off[ep + 1]]
            f = flags[ep]
            rest = range(o.placed_step, o.removed_step)
            # the owner's trades on the side opposite to this order (trade_dir = -side)
            opp = [s for s in rest if X[s, TRADE] == -o.side]
            if not opp:   # its only opposite trade fell on the removal step, outside the resting window
                continue
            first = opp[0]
            for s in rest:
                phase = "trade" if s in opp else ("before" if s < first else "after")
                phase_hits[name][phase][0] += int(f[s])
                phase_hits[name][phase][1] += 1
            caught = bool(f[o.placed_step:o.removed_step].any())
            k = len(opp)
            by_trades[k if k < 5 else 5][0] += caught
            by_trades[k if k < 5 else 5][1] += 1
            wait = first - o.placed_step
            wb = "0" if wait == 0 else "1-4" if wait < 5 else "5-9" if wait < 10 else "10+"
            by_wait[wb][0] += caught
            by_wait[wb][1] += 1
            hit = np.flatnonzero(f[o.placed_step:o.removed_step])
            if hit.size:
                first_flag_vs_trade.append(int(hit[0] + o.placed_step - first))

    print(f"Watchdog {args.tag}, {args.split} split, manipulative orders only\n")
    print(f"{'source':18s} {'phase':7s} {'steps':>7s} {'flag rate':>9s}")
    for name, phases in phase_hits.items():
        for ph in ("before", "trade", "after"):
            fl, n = phases[ph]
            print(f"{name:18s} {ph:7s} {n:7d} {fl / n if n else float('nan'):9.3f}")
    print("\norder recall by number of opposite trades while resting (5 = 5 or more)")
    for k in sorted(by_trades):
        c, n = by_trades[k]
        print(f"  {k}: {c}/{n} = {c / n:.3f}")
    print("\norder recall by wait before the first opposite trade (agent steps of 10 events)")
    for k in ("0", "1-4", "5-9", "10+"):
        if k in by_wait:
            c, n = by_wait[k]
            print(f"  {k:4s}: {c}/{n} = {c / n:.3f}")
    d = np.array(first_flag_vs_trade)
    if d.size:
        print(f"\nfirst flag minus first opposite trade (steps), caught orders: median {np.median(d):.0f}, "
              f"share flagged before any trade {np.mean(d < 0):.3f}, on the trade step {np.mean(d == 0):.3f}, "
              f"after {np.mean(d > 0):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
