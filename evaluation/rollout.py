"""Roll out one policy for one episode and record what surveillance can observe.

Returns episode PnL, the lifecycle of every large order (`OrderRecord`), and optionally a per-step
Watchdog observation and label:

  observation (46) = 40 normalized book features as the market sees them (resting large orders
                     included in displayed size) + 6 features of the monitored participant:
                     [log1p(resting size / depth), oldest resting age / 200 (<=5),
                      log1p(placed this step / depth), log1p(cancelled this step / depth),
                      trade direction this step (+1 buy, -1 sell, 0 none),
                      signed log1p(|resting bid - resting ask| / depth)]
  label            = a manipulative order (its owner traded on the opposite side while it rested)
                     is resting at this step. Assigned with hindsight after the episode.
"""
from __future__ import annotations

import numpy as np

from env.lob_env import BUY, LimitOrderBookEnv
from evaluation.baseline_detector import OrderRecord

BOOK_DIM = 40
PARTICIPANT_DIM = 6
OBS_DIM = BOOK_DIM + PARTICIPANT_DIM


def participant_features(env: LimitOrderBookEnv, placed: float, cancelled: float, trade_dir: int) -> np.ndarray:
    t = env.t
    depth = max(float(env.stats.touch_depth[t]), 1.0)
    bid = sum(s.size for s in env.spoofs if s.side > 0)
    ask = sum(s.size for s in env.spoofs if s.side < 0)
    age = max((t - s.placed_t for s in env.spoofs), default=0)
    imbalance = bid - ask
    return np.array([
        np.log1p((bid + ask) / depth),
        min(age / 200.0, 5.0),
        np.log1p(placed / depth),
        np.log1p(cancelled / depth),
        float(trade_dir),
        np.sign(imbalance) * np.log1p(abs(imbalance) / depth),
    ], dtype=np.float32)


def record_episode(env: LimitOrderBookEnv, policy, rng, seed: int, features: bool = True) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset(rng)
    live: dict[int, tuple] = {}   # id(spoof) -> (spoof, OrderRecord); holding the object keeps id unique
    done: list[OrderRecord] = []
    X, live_ids = [], []
    next_id = step = trades = manip_trades = 0
    while True:
        a = policy.act(obs, env, rng)
        t = env.t
        pre = {id(s): s for s in env.spoofs}
        obs, _, term, trunc, info = env.step(a)

        trade_dir = 0
        if info["trade_price"] is not None:
            trades += 1
            trade_dir = 1 if a == BUY else -1
            against = [k for k, s in pre.items() if s.side == -trade_dir and k in live]
            for k in against:
                live[k][1].opposite_trades += 1
            manip_trades += bool(against)

        post = {id(s): s for s in env.spoofs}
        placed = 0.0
        for k, s in post.items():
            if k not in live:
                live[k] = (s, OrderRecord(next_id, s.side, s.size, s.size / env.stats.touch_depth[t], step,
                                          events_per_step=env.cfg.events_per_step))
                next_id += 1
                placed += s.size
        cancelled = 0.0
        run_over = {(f["side"], f["price"]) for f in info["run_over"]}
        for k in [k for k in live if k not in post]:
            s, rec = live.pop(k)
            rec.removed_step = step
            rec.removed_by = "run_over" if (s.side, s.price) in run_over else ("episode_end" if trunc else "cancel")
            if rec.removed_by == "cancel":
                cancelled += s.size
            done.append(rec)

        if features:
            X.append(np.concatenate([obs[:BOOK_DIM], participant_features(env, placed, cancelled, trade_dir)]))
            live_ids.append([rec.order_id for _, rec in live.values()])
        step += 1
        if term or trunc:
            break

    out = {"pnl": float(info["pnl"]), "orders": done, "trades": trades, "manip_trades": manip_trades, "steps": step}
    if features:
        manipulative = {r.order_id for r in done if r.manipulative}
        out["X"] = np.asarray(X, dtype=np.float32)
        out["y"] = np.array([any(i in manipulative for i in ids) for ids in live_ids], dtype=bool)
    return out
