"""Cross-stock scale normalization.

Raw prices and share counts are not comparable across tickers (AAPL ~$585 with a ~60-tick
spread vs INTC ~$27 with a 1-tick spread), and tick offsets are not comparable either.
So the encoding is:

  price feature = (level price - unimpacted mid) / trailing median spread     [spread units]
  size feature  = log1p(level size / trailing EWMA size at that level & side)

All reference statistics are strictly causal (computed from events <= t).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from env.lobster_data import LobsterDay

PRICE_CLIP = 20.0      # empty/deep levels are pinned to +-PRICE_CLIP spread units
SIZE_CLIP = 10.0
AGENT_CLIP = 5.0
SPREAD_WINDOW = 5_000  # events
SIZE_HALFLIFE = 5_000  # events


@dataclass
class ReferenceStats:
    ref_spread: np.ndarray    # (N,) trailing median touch spread, dollars
    ask_ref_size: np.ndarray  # (N, L) trailing EWMA size per ask level
    bid_ref_size: np.ndarray  # (N, L)

    @cached_property
    def touch_depth(self) -> np.ndarray:
        """(N,) trailing mean depth at the touch, averaged over both sides."""
        return (self.ask_ref_size[:, 0] + self.bid_ref_size[:, 0]) / 2.0


def reference_stats(day: LobsterDay, spread_window: int = SPREAD_WINDOW,
                    size_halflife: int = SIZE_HALFLIFE) -> ReferenceStats:
    spread = pd.Series(day.spread)
    ref_spread = spread.rolling(spread_window, min_periods=1).median().ffill().to_numpy()
    fallback = np.nanmedian(day.spread)
    ref_spread = np.where(np.isfinite(ref_spread) & (ref_spread > 0), ref_spread, fallback)

    def ewma(sizes: np.ndarray) -> np.ndarray:
        # empty levels (size 0) are excluded from the average rather than dragging it to zero
        frame = pd.DataFrame(np.where(sizes > 0, sizes, np.nan))
        out = frame.ewm(halflife=size_halflife, adjust=False, ignore_na=True).mean().ffill()
        return out.fillna(1.0).clip(lower=1.0).to_numpy()

    return ReferenceStats(ref_spread, ewma(day.ask_size), ewma(day.bid_size))


def encode_book(day: LobsterDay, stats: ReferenceStats, idx=slice(None),
                ask_size=None, bid_size=None) -> np.ndarray:
    """Encode book rows `idx` into (..., 4L) float32 in LOBSTER order
    [ask_p_1, ask_s_1, bid_p_1, bid_s_1, ask_p_2, ...].

    `ask_size`/`bid_size` optionally override the historical sizes (e.g. to add a resting spoof).
    """
    ask_p, bid_p = day.ask_price[idx], day.bid_price[idx]
    ask_s = day.ask_size[idx] if ask_size is None else ask_size
    bid_s = day.bid_size[idx] if bid_size is None else bid_size
    mid = day.mid[idx]
    spr = stats.ref_spread[idx]
    if np.ndim(mid):
        mid, spr = mid[..., None], spr[..., None]

    ask_pf = np.nan_to_num((ask_p - mid) / spr, nan=PRICE_CLIP)
    bid_pf = np.nan_to_num((bid_p - mid) / spr, nan=-PRICE_CLIP)
    ask_sf = np.log1p(ask_s / stats.ask_ref_size[idx])
    bid_sf = np.log1p(bid_s / stats.bid_ref_size[idx])

    out = np.stack([
        np.clip(ask_pf, -PRICE_CLIP, PRICE_CLIP),
        np.clip(ask_sf, 0.0, SIZE_CLIP),
        np.clip(bid_pf, -PRICE_CLIP, PRICE_CLIP),
        np.clip(bid_sf, 0.0, SIZE_CLIP),
    ], axis=-1)
    return out.reshape(*out.shape[:-2], -1).astype(np.float32)


def encode_agent(inventory: float, cash_pnl: float, spoof_size: float, *, mid: float,
                 lot: int, max_inventory: int, touch_depth: float) -> np.ndarray:
    """Scale-free agent-internal features, clipped to [-AGENT_CLIP, AGENT_CLIP].

    inventory  -> fraction of the inventory limit
    cash_pnl   -> PnL in units of "one lot's notional", so $1 on AAPL and $1 on INTC are not equated
    spoof_size -> multiple of trailing touch depth
    """
    feats = np.array([
        inventory / max_inventory,
        cash_pnl / (mid * lot) if mid > 0 else 0.0,
        spoof_size / max(touch_depth, 1.0),
    ], dtype=np.float64)
    return np.clip(feats, -AGENT_CLIP, AGENT_CLIP).astype(np.float32)
