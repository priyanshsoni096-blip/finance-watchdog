"""Price-impact layer on top of historical replay.

Historical LOBSTER prices cannot react to an injected order, so a large visible resting order
shifts the *impacted* mid used for the agent's fills:

  linear: shift = lambda * ref_spread * (net_resting / touch_depth)
  sqrt:   shift = lambda * ref_spread * sign(net) * sqrt(|net| / touch_depth)

`net_resting` = resting spoof buy shares - resting spoof sell shares (a big bid pushes price up).
Lambda is calibrated from data (see `calibrate_lambda`), not hand-picked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.lobster_data import NEW, LobsterDay
from env.normalization import ReferenceStats


@dataclass
class PriceImpact:
    lam: float
    form: str = "linear"  # "linear" | "sqrt"

    def shift(self, net_resting: float, touch_depth: float, ref_spread: float) -> float:
        if self.lam == 0.0 or net_resting == 0.0:
            return 0.0
        x = net_resting / max(touch_depth, 1.0)
        if self.form == "linear":
            g = x
        elif self.form == "sqrt":
            g = np.sign(x) * np.sqrt(abs(x))
        else:
            raise ValueError(f"unknown impact form {self.form!r}")
        return float(self.lam * ref_spread * g)


def calibration_targets(day: LobsterDay, stats: ReferenceStats) -> dict[str, float]:
    """Real-data statistics the impact coefficient is calibrated against."""
    new_sizes = day.size[day.event_type == NEW]
    dmid = np.abs(np.diff(day.mid))
    moves = dmid[np.isfinite(dmid) & (dmid > 0)]
    return {
        "p95_new_order_size": float(np.percentile(new_sizes, 95)),
        "median_touch_depth": float(np.median(stats.touch_depth)),
        "median_ref_spread": float(np.median(stats.ref_spread)),
        "median_nonzero_mid_move": float(np.median(moves)),
    }


def percentile_rule_lambda(day: LobsterDay, stats: ReferenceStats, form: str = "linear") -> float:
    """REJECTED calibration rule, kept to document the negative result.

    Sets lambda so an order at the 95th percentile of new-order sizes moves the mid by the median
    non-zero one-event move. On thick 1-tick books (MSFT, INTC) p95 order size is ~0.09x touch
    depth, so lambda is inflated and a 10x-depth spoof extrapolates to a 44-58 spread move.
    """
    t = calibration_targets(day, stats)
    unit = PriceImpact(1.0, form).shift(t["p95_new_order_size"], t["median_touch_depth"], t["median_ref_spread"])
    return float(t["median_nonzero_mid_move"] / unit)


def order_flow_imbalance(day: LobsterDay) -> np.ndarray:
    """Per-event order-flow imbalance at the touch (Cont, Kukanov & Stoikov 2014), in shares."""
    pb, qb, pa, qa = day.bid_price[:, 0], day.bid_size[:, 0], day.ask_price[:, 0], day.ask_size[:, 0]
    e = np.zeros(len(day))
    with np.errstate(invalid="ignore"):
        e[1:] = ((pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
                 - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1])
    return np.nan_to_num(e)


def calibrate_lambda(day: LobsterDay, stats: ReferenceStats, form: str = "linear",
                     window: int = 200, warmup: int = 5_000) -> tuple[float, dict]:
    """Fit lambda from real order flow: over non-overlapping windows of `window` events,
    regress mid change (in trailing spreads) on g(OFI / touch depth) through the origin,
    where g is the impact form. The slope is exactly PriceImpact.lam."""
    cum = np.cumsum(order_flow_imbalance(day))
    idx = np.arange(warmup, len(day) - window, window)
    x = (cum[idx + window] - cum[idx]) / np.maximum(stats.touch_depth[idx], 1.0)
    y = (day.mid[idx + window] - day.mid[idx]) / stats.ref_spread[idx]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    g = x if form == "linear" else np.sign(x) * np.sqrt(np.abs(x))
    lam = float((g @ y) / (g @ g))
    r2 = float(1 - np.sum((y - lam * g) ** 2) / np.sum((y - y.mean()) ** 2))
    t = calibration_targets(day, stats)
    t.update({"ofi_window": window, "ofi_r2": r2, "ofi_samples": int(ok.sum()),
              "percentile_rule_lambda": percentile_rule_lambda(day, stats, form)})
    return lam, t
