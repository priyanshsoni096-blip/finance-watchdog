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

`StepTracker` holds the per-step tracking so offline recording and online surveillance (a Watchdog acting
inside the episode) build exactly the same observation.
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


class StepTracker:
    """Tracks one participant's large orders across env steps and builds the 46-dim surveillance observation.

    Call `before_step()` just before `env.step(action)` and `after_step(...)` right after it.
    `force_cancel(step)` removes resting orders on behalf of surveillance; those orders are recorded with
    removed_by="intervention", so they are never mistaken for the participant's own cancels.
    """

    def __init__(self, env: LimitOrderBookEnv):
        self.env = env
        self.live: dict[int, tuple] = {}   # id(spoof) -> (spoof, OrderRecord); holding the object keeps id unique
        self.done: list[OrderRecord] = []
        self.next_id = 0
        self.trades = 0
        self.manip_trades = 0
        self.last_removed: list[OrderRecord] = []
        self._pre: dict[int, object] = {}
        self._t = env.t

    def before_step(self) -> None:
        self._pre = {id(s): s for s in self.env.spoofs}
        self._t = self.env.t

    def after_step(self, obs: np.ndarray, action: int, info: dict, step: int, truncated: bool) -> np.ndarray:
        env, t, pre = self.env, self._t, self._pre
        trade_dir = 0
        if info["trade_price"] is not None:
            self.trades += 1
            trade_dir = 1 if action == BUY else -1
            against = [k for k, s in pre.items() if s.side == -trade_dir and k in self.live]
            for k in against:
                self.live[k][1].opposite_trades += 1
            self.manip_trades += bool(against)

        post = {id(s): s for s in env.spoofs}
        placed = 0.0
        for k, s in post.items():
            if k not in self.live:
                self.live[k] = (s, OrderRecord(self.next_id, s.side, s.size, s.size / env.stats.touch_depth[t], step,
                                               events_per_step=env.cfg.events_per_step))
                self.next_id += 1
                placed += s.size
        cancelled = 0.0
        self.last_removed = []
        run_over = {(f["side"], f["price"]) for f in info["run_over"]}
        for k in [k for k in self.live if k not in post]:
            s, rec = self.live.pop(k)
            rec.removed_step = step
            rec.removed_by = "run_over" if (s.side, s.price) in run_over else ("episode_end" if truncated else "cancel")
            if rec.removed_by == "cancel":
                cancelled += s.size
            self.done.append(rec)
            self.last_removed.append(rec)
        return np.concatenate([obs[:BOOK_DIM], participant_features(env, placed, cancelled, trade_dir)])

    def live_order_ids(self) -> list[int]:
        return [rec.order_id for _, rec in self.live.values()]

    def force_cancel(self, step: int) -> int:
        """Surveillance cancels every resting order of the participant. Returns how many were cancelled."""
        n = len(self.env.spoofs)
        for k in list(self.live):
            _, rec = self.live.pop(k)
            rec.removed_step = step
            rec.removed_by = "intervention"
            self.done.append(rec)
        self.env.spoofs.clear()
        return n


def record_episode(env: LimitOrderBookEnv, policy, rng, seed: int, features: bool = True) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset(rng)
    tracker = StepTracker(env)
    X, live_ids = [], []
    step = 0
    while True:
        a = policy.act(obs, env, rng)
        tracker.before_step()
        obs, _, term, trunc, info = env.step(a)
        x = tracker.after_step(obs, a, info, step, trunc)
        if features:
            X.append(x)
            live_ids.append(tracker.live_order_ids())
        step += 1
        if term or trunc:
            break

    out = {"pnl": float(info["pnl"]), "orders": tracker.done, "trades": tracker.trades,
           "manip_trades": tracker.manip_trades, "steps": step}
    if features:
        manipulative = {r.order_id for r in tracker.done if r.manipulative}
        out["X"] = np.asarray(X, dtype=np.float32)
        out["y"] = np.array([any(i in manipulative for i in ids) for ids in live_ids], dtype=bool)
    return out
