"""Rule-based baseline spoofing detector (handoff §5.5).

Rule: flag a visible order if
    size >= N x trailing touch depth at placement
    AND it is cancelled within M events
    AND it did not execute before the cancel.

The same rule runs on two front-ends:
  * simulated orders placed by agents in LimitOrderBookEnv (`OrderRecord`)
  * real LOBSTER message streams (`real_data_flags`), where no labels exist — so the result is a
    flag rate on unlabeled real flow, not a false-positive rate.

Labels for simulated orders use an economic proxy for intent: an order is manipulative if the same
agent traded on the OPPOSITE side while it rested (spoof bid -> sell into the raised price).
A large order that is placed and cancelled with no such trade is labelled legitimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from env.lobster_data import DELETE, EXEC_VISIBLE, NEW, LobsterDay
from env.normalization import ReferenceStats


@dataclass
class OrderRecord:
    order_id: int
    side: int                 # +1 bid, -1 ask
    size: float
    depth_mult: float         # size / trailing touch depth at placement
    placed_step: int
    removed_step: int | None = None
    removed_by: str | None = None   # "cancel" | "run_over" | "episode_end"
    opposite_trades: int = 0
    events_per_step: int = 1        # market events per agent step when the order was recorded

    @property
    def lifetime(self) -> int | None:
        """Lifetime in agent steps."""
        return None if self.removed_step is None else self.removed_step - self.placed_step

    @property
    def lifetime_events(self) -> int | None:
        """Lifetime in market events — the unit the rule's M uses, on simulated and real data alike."""
        return None if self.lifetime is None else self.lifetime * self.events_per_step

    @property
    def manipulative(self) -> bool:
        return self.opposite_trades > 0


@dataclass(frozen=True)
class RuleDetector:
    n_mult: float
    m_events: int

    def flags(self, order: OrderRecord) -> bool:
        return (order.depth_mult >= self.n_mult
                and order.removed_by == "cancel"
                and order.lifetime_events is not None
                and order.lifetime_events <= self.m_events)


def classification_metrics(orders: list[OrderRecord], det: RuleDetector) -> dict:
    tp = fp = fn = tn = 0
    for o in orders:
        flagged, positive = det.flags(o), o.manipulative
        if flagged and positive:
            tp += 1
        elif flagged:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else nan
    recall = tp / (tp + fn) if tp + fn else nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else nan)
    fpr = fp / (fp + tn) if fp + tn else nan
    return {"orders": len(orders), "positives": tp + fn, "negatives": fp + tn, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def tune(orders: list[OrderRecord], n_grid=(2, 5, 8), m_grid=(10, 50, 200, 1000)) -> tuple[RuleDetector, list]:
    """Pick (N, M) with the best F1 on the given (training-pool) orders."""
    scored = []
    for n in n_grid:
        for m in m_grid:
            det = RuleDetector(n, m)
            met = classification_metrics(orders, det)
            f1 = met["f1"] if np.isfinite(met["f1"]) else -1.0
            scored.append((f1, -m, det, met))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2], [(d.n_mult, d.m_events, m) for _, _, d, m in scored]


def real_data_flags(day: LobsterDay, stats: ReferenceStats, det: RuleDetector, warmup: int = 30_000) -> dict:
    """Apply the rule to real NEW -> DELETE message pairs. Orders with a visible execution before
    their deletion are not flagged (partially filled then cancelled is good-faith per CFTC guidance)."""
    idx = np.arange(len(day))
    et, oid = day.event_type, day.order_id

    def first(mask):
        s = pd.Series(idx[mask], index=oid[mask])
        return s.groupby(level=0).min()

    df = pd.DataFrame({"placed": first(et == NEW)})
    df = df[df.placed >= warmup]
    placed = df.placed.to_numpy()
    df["depth_mult"] = day.size[placed] / stats.touch_depth[placed]
    df["deleted"] = first(et == DELETE).reindex(df.index)
    df["first_exec"] = first(et == EXEC_VISIBLE).reindex(df.index)

    big = df[df.depth_mult >= det.n_mult]
    fast_cancel = big.deleted.notna() & ((big.deleted - big.placed) <= det.m_events)
    no_exec = big.first_exec.isna() | (big.first_exec > big.deleted)
    flags = int((fast_cancel & no_exec).sum())
    events = len(day) - warmup
    return {"events": int(events), "new_orders": int(len(df)), "large_orders": int(len(big)),
            "flags": flags, "flags_per_10k_events": 1e4 * flags / events}
