"""Build and load Watchdog datasets from frozen agents.

The Spoofer pool is frozen before the Watchdog trains (no co-evolution). Every source policy is rolled
out once in LimitOrderBookEnv and its surveillance view is saved, so Watchdog training replays fixed
trajectories and cannot influence the agents it watches.

Splits:
  train    SPOOFER-01..04 on their own stocks, HONEST and FLICKER on AAPL/MSFT/GOOG/INTC
  test     the same sources with different seeds (in-distribution check)
  heldout  never used for training: SPOOFER-05 (AMZN), SCRIPTED-ATK on all stocks,
           HONEST and FLICKER on AMZN

Usage:
    python watchdog/dataset.py --episodes 30 [--tag main]
Writes data/watchdog/<split>/<SOURCE>_<TICKER>.npz
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from env.lob_env import EnvConfig, LimitOrderBookEnv  # noqa: E402
from env.lobster_data import load_day  # noqa: E402
from env.normalization import reference_stats  # noqa: E402
from evaluation.agents import Flicker, Honest, ModelPolicy, ScriptedSpoof  # noqa: E402
from evaluation.baseline_detector import OrderRecord  # noqa: E402
from evaluation.rollout import OBS_DIM, record_episode  # noqa: E402

DATA_DIR = ROOT / "data" / "watchdog"
TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]
HELD_OUT = "AMZN"
SEED_BASE = {"train": 100_000, "test": 200_000, "heldout": 300_000}
REMOVED = {"cancel": 0, "run_over": 1, "episode_end": 2}
REMOVED_INV = {v: k for k, v in REMOVED.items()}
ORDER_COLS = ["episode", "order_id", "side", "size", "depth_mult", "placed", "removed", "removed_by", "opposite_trades"]


@dataclass
class Source:
    name: str
    ticker: str
    make_policy: object
    cfg: dict


def sources(tag: str = "main") -> dict[str, list[Source]]:
    from train_spoofer import AGENTS
    train, heldout = [], []
    for agent, spec in AGENTS.items():
        path = ROOT / "checkpoints" / agent / tag / "model.zip"
        if not path.exists():
            print(f"[skip] {agent}: no checkpoint at {path.relative_to(ROOT)}")
            continue
        src = Source(agent, spec["ticker"], lambda p=path: ModelPolicy(p), spec["cfg"])
        (heldout if spec["ticker"] == HELD_OUT else train).append(src)
    for tk in TICKERS:
        bucket = heldout if tk == HELD_OUT else train
        bucket.append(Source("HONEST", tk, Honest, {}))
        bucket.append(Source("FLICKER", tk, Flicker, {}))
        heldout.append(Source("SCRIPTED-ATK", tk, ScriptedSpoof, {}))
    return {"train": train, "test": train, "heldout": heldout}


def _orders_to_array(orders_by_ep: list[list[OrderRecord]]) -> np.ndarray:
    rows = [[ep, o.order_id, o.side, o.size, o.depth_mult, o.placed_step, o.removed_step,
             REMOVED[o.removed_by], o.opposite_trades]
            for ep, orders in enumerate(orders_by_ep) for o in orders]
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(ORDER_COLS))


def orders_from_array(arr: np.ndarray) -> list[tuple[int, OrderRecord]]:
    """-> [(episode, OrderRecord)]"""
    out = []
    for r in arr:
        rec = OrderRecord(int(r[1]), int(r[2]), float(r[3]), float(r[4]), int(r[5]), int(r[6]),
                          REMOVED_INV[int(r[7])], int(r[8]))
        out.append((int(r[0]), rec))
    return out


def build_source(src: Source, split: str, episodes: int, cache: dict) -> dict:
    if src.ticker not in cache:
        d = load_day(src.ticker)
        cache[src.ticker] = (d, reference_stats(d))
    d, s = cache[src.ticker]
    env = LimitOrderBookEnv(EnvConfig(ticker=src.ticker, **src.cfg), day=d, stats=s)
    policy = src.make_policy()
    salt = zlib.crc32(f"{src.name}/{src.ticker}".encode()) % 10_000
    rng = np.random.default_rng(SEED_BASE[split] + salt)
    eps = [record_episode(env, policy, rng, seed=SEED_BASE[split] + salt * 7 + 97 * j) for j in range(episodes)]
    lengths = np.array([len(e["y"]) for e in eps])
    return {
        "X": np.concatenate([e["X"] for e in eps]),
        "y": np.concatenate([e["y"] for e in eps]),
        "ep_offsets": np.concatenate([[0], np.cumsum(lengths)]),
        "orders": _orders_to_array([e["orders"] for e in eps]),
        "pnl": np.array([e["pnl"] for e in eps]),
    }


def load_split(split: str, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """-> {"SOURCE_TICKER": {"X", "y", "ep_offsets", "orders", "pnl"}}"""
    out = {}
    for f in sorted((data_dir / split).glob("*.npz")):
        with np.load(f) as z:
            out[f.stem] = {k: z[k] for k in z.files}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30, help="train episodes per source; test/heldout use a third (min 10)")
    p.add_argument("--eval-episodes", type=int, default=None, help="override test/heldout episodes per source")
    p.add_argument("--tag", default="main")
    p.add_argument("--splits", nargs="*", default=["train", "test", "heldout"])
    p.add_argument("--out", default=str(DATA_DIR))
    args = p.parse_args()

    t0, cache = time.time(), {}
    by_split = sources(args.tag)
    for split in args.splits:
        if split == "train":
            n = args.episodes
        else:
            n = args.eval_episodes if args.eval_episodes is not None else max(10, args.episodes // 3)
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in by_split[split]:
            data = build_source(src, split, n, cache)
            np.savez_compressed(out_dir / f"{src.name}_{src.ticker}.npz", **data)
            assert data["X"].shape[1] == OBS_DIM
            print(f"[{time.time() - t0:6.0f}s] {split:7s} {src.name:12s} {src.ticker} episodes={n} "
                  f"steps={len(data['y'])} positive_steps={data['y'].mean():.3f} orders={len(data['orders'])}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
