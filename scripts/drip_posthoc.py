"""POST-HOC analysis of the pre-registered DRIP-ATK result (run after the verdict was fixed; informs no verdict).

Usage:
    python scripts/drip_posthoc.py [--tag causal_g0_1m]

For every DRIP-ATK opposite trade against its resting order (the act), reports the Watchdog's flag rate split by
stock, by whether a same-side camouflage trade happened earlier while the order rested, and by whether it is the
order's first opposite trade, and by the order's age (steps since placement) at the act. Reads data/watchdog/drip (written by watchdog/drip_test.py). Prints to the terminal.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from watchdog.dataset import TRADE_COL, load_split, orders_from_array  # noqa: E402
from watchdog.watchdog_env import flag_episode, load_obs_norm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="causal_g0_1m")
    args = p.parse_args()
    from sb3_contrib import RecurrentPPO
    path = ROOT / "checkpoints" / "WATCHDOG" / args.tag / "model.zip"
    model, obs_norm = RecurrentPPO.load(path, device="cpu"), load_obs_norm(path.parent)
    stats = defaultdict(lambda: [0, 0])
    for name, src in sorted(load_split("drip").items()):
        if not name.startswith("DRIP"):
            continue
        tk = name.split("_")[1]
        off = src["ep_offsets"]
        flags = [flag_episode(model, src["X"][a:b], obs_norm) for a, b in zip(off[:-1], off[1:])]
        for ep, o in orders_from_array(src["orders"]):
            X, f = src["X"][off[ep]:off[ep + 1]], flags[ep]
            seen_same = seen_opp = False
            for s in range(o.placed_step, o.removed_step):
                d = X[s, TRADE_COL]
                if d == o.side:
                    seen_same = True
                elif d == -o.side:
                    age = s - o.placed_step
                    age_bucket = "1-4" if age < 5 else "5-9" if age < 10 else "10-19" if age < 20 else "20+"
                    for key in (("stock", tk), ("camouflage before", seen_same), ("first act", not seen_opp),
                                ("age (steps)", age_bucket), (f"age {tk}", age_bucket)):
                        stats[key][0] += int(f[s])
                        stats[key][1] += 1
                    seen_opp = True
    for key in sorted(stats, key=str):
        h, n = stats[key]
        print(f"{key[0]:18s} {str(key[1]):6s} acts={n:5d} flagged={h / n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
