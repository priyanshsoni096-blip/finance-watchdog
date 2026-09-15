"""Synthetic agent-based market: prices emerge from order matching between simple background traders.

Evaluation only (sim-to-sim robustness, design §4.3). Two background populations act one event at a time:

  noise traders        place limit orders a geometric number of ticks behind the touch (sometimes inside the
                       spread) or send a small market order; sizes are log-normal. Every resting noise order is
                       cancelled with a constant per-event hazard (zero-intelligence style), so the cancel rate
                       scales with book size and liquidity reaches a steady state. A first version cancelled one
                       order per 35% of events regardless of book size; liquidity then grew without bound (median
                       touch depth 88,770 shares, mid moving on 0.0% of events).
  imbalance followers  compare visible bid and ask volume over the top levels and send a market order toward the
                       heavier side when the imbalance exceeds a threshold — the channel through which a visible
                       spoof can move the price without any impact formula

A side that empties is refilled near the last mid so the book never stays one-sided. Everything is seeded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthetic.market import BUY, SELL, OrderBook


@dataclass
class MarketParams:
    start_price: int = 5_000            # ticks (e.g. $50.00 at a $0.01 tick)
    tick_value: float = 0.01            # dollars per tick
    init_levels: int = 10
    init_size: int = 500
    size_log_mean: float = float(np.log(300))
    size_log_sd: float = 0.8
    market_size_log_mean: float = float(np.log(100))   # real MSFT/INTC median execution size is 100 shares
    market_size_log_sd: float = 1.2
    cancel_hazard: float = 0.002        # per resting noise order, per event
    p_follower: float = 0.15            # share of arrival events taken by an imbalance follower
    noise_p_market: float = 0.1         # share of noise arrivals that are market orders (rest are limit orders)
    noise_inside_spread: float = 0.3
    noise_depth_geom_p: float = 0.35    # ticks behind the touch ~ Geometric(p) - 1
    follower_levels: int = 3
    follower_threshold: float = 0.35
    follower_p_react: float = 0.6


class SyntheticMarket:
    NOISE, FOLLOWER = "noise", "follower"

    def __init__(self, params: MarketParams | None = None, seed: int = 0):
        self.p = params or MarketParams()
        self.rng = np.random.default_rng(seed)
        self.book = OrderBook()
        self.noise_orders: list[int] = []
        self.t = 0
        self.counts = {"limit": 0, "cancel": 0, "market": 0, "follower_market": 0, "fills": 0}
        self.last_mid = float(self.p.start_price)
        for i in range(1, self.p.init_levels + 1):
            self._rest(BUY, self.p.start_price - i, self.p.init_size)
            self._rest(SELL, self.p.start_price + i, self.p.init_size)

    # ------------------------------------------------------------ helpers
    def _size(self, log_mean: float, log_sd: float | None = None) -> int:
        sd = self.p.size_log_sd if log_sd is None else log_sd
        return max(1, int(round(self.rng.lognormal(log_mean, sd))))

    def _market_size(self) -> int:
        return self._size(self.p.market_size_log_mean, self.p.market_size_log_sd)

    def _rest(self, side: int, price: int, size: int) -> None:
        oid, _ = self.book.limit(side, price, size, self.NOISE)
        if oid is not None:
            self.noise_orders.append(oid)

    def imbalance(self, levels: int | None = None) -> float:
        n = levels or self.p.follower_levels
        bid = sum(s for _, s in self.book.levels(BUY, n))
        ask = sum(s for _, s in self.book.levels(SELL, n))
        return 0.0 if bid + ask == 0 else (bid - ask) / (bid + ask)

    def _refill(self) -> None:
        mid = int(round(self.last_mid))
        if self.book.best_bid() is None:
            self._rest(BUY, mid - 1, self.p.init_size)
        if self.book.best_ask() is None:
            self._rest(SELL, mid + 1, self.p.init_size)

    # ------------------------------------------------------------ agents
    def _noise_arrival(self) -> list:
        side = BUY if self.rng.random() < 0.5 else SELL
        if self.rng.random() < self.p.noise_p_market:
            self.counts["market"] += 1
            return self.book.market(side, self._market_size(), self.NOISE)
        self.counts["limit"] += 1
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if ba - bb > 1 and self.rng.random() < self.p.noise_inside_spread:
            price = bb + 1 if side == BUY else ba - 1
        else:
            behind = int(self.rng.geometric(self.p.noise_depth_geom_p)) - 1
            price = bb - behind if side == BUY else ba + behind
        oid, fills = self.book.limit(side, price, self._size(self.p.size_log_mean), self.NOISE)
        if oid is not None:
            self.noise_orders.append(oid)
        return fills

    def _cancel_event(self) -> list:
        while self.noise_orders:
            k = int(self.rng.integers(len(self.noise_orders)))
            oid = self.noise_orders[k]
            self.noise_orders[k] = self.noise_orders[-1]
            self.noise_orders.pop()
            if self.book.cancel(oid) > 0:
                break
        return []

    def _prune(self) -> None:
        """Drop ids of noise orders that were filled, so the cancel rate uses an accurate count."""
        self.noise_orders = [oid for oid in self.noise_orders if oid in self.book.orders]

    def _follower_event(self) -> list | None:
        """A market order toward the heavier side, or None when the follower does not act."""
        imb = self.imbalance()
        if abs(imb) < self.p.follower_threshold or self.rng.random() > self.p.follower_p_react:
            return None
        side = BUY if imb > 0 else SELL      # heavy bids read as buying interest -> buy
        self.counts["follower_market"] += 1
        return self.book.market(side, self._market_size(), self.FOLLOWER)

    # ------------------------------------------------------------ loop
    def step(self) -> list:
        """One background event. Returns its fills (so fills against an injected order can be booked).

        Event type is drawn by rate: cancellations at cancel_hazard x (resting noise orders) against arrivals at
        rate 1, so each cancellation is its own event, as in the LOBSTER message stream."""
        if self.t % 500 == 0:
            self._prune()
        # An agent acting between events (synthetic_env.py) can empty a side of the book. On its own the market
        # ends every step two-sided, so this refill is a no-op there and calibration results are unchanged.
        self._refill()
        r_cancel = self.p.cancel_hazard * len(self.noise_orders)
        if self.rng.random() < r_cancel / (r_cancel + 1.0):
            self.counts["cancel"] += 1
            fills = self._cancel_event()
        else:
            fills = self._follower_event() if self.rng.random() < self.p.p_follower else None
            if fills is None:                # an idle follower is not an order event; a noise trader acts
                fills = self._noise_arrival()
        self.counts["fills"] += len(fills)
        self._refill()
        mid = self.book.mid()
        if mid is not None:
            self.last_mid = mid
        self.t += 1
        return fills

    def run(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Advance n events; returns (mid, spread) per event in ticks."""
        mids, spreads = np.empty(n), np.empty(n)
        for i in range(n):
            self.step()
            mids[i] = self.last_mid
            spreads[i] = self.book.best_ask() - self.book.best_bid()
        return mids, spreads
