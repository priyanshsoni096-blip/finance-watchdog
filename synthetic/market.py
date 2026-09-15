"""Minimal price-time-priority limit order book for the synthetic agent-based market.

Used for evaluation only (sim-to-sim robustness): no agent is ever trained in it. Prices are integer ticks.
Each price level is a FIFO queue; cancellations are lazy (the order's remaining size is set to 0 and skipped by
matching), so cancel is O(1). Every fill reports the resting order's owner, so a spoof that is hit by the market
can be booked to whoever placed it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

BUY, SELL = 1, -1


@dataclass
class Order:
    order_id: int
    side: int
    price: int
    size: int
    owner: str


@dataclass
class Fill:
    price: int
    size: int
    maker_owner: str
    maker_id: int
    taker_owner: str
    taker_side: int


@dataclass
class OrderBook:
    bids: dict[int, deque] = field(default_factory=dict)   # price -> deque[Order]
    asks: dict[int, deque] = field(default_factory=dict)
    bid_volume: dict[int, int] = field(default_factory=dict)
    ask_volume: dict[int, int] = field(default_factory=dict)
    orders: dict[int, Order] = field(default_factory=dict)  # live orders by id
    _next_id: int = 1

    # ------------------------------------------------------------ queries
    def best_bid(self) -> int | None:
        return max(self.bid_volume) if self.bid_volume else None

    def best_ask(self) -> int | None:
        return min(self.ask_volume) if self.ask_volume else None

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (a + b) / 2.0

    def levels(self, side: int, n: int) -> list[tuple[int, int]]:
        """Top `n` (price, total size) levels, best first."""
        vol = self.bid_volume if side == BUY else self.ask_volume
        prices = sorted(vol, reverse=(side == BUY))[:n]
        return [(p, vol[p]) for p in prices]

    def snapshot(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(ask_price, ask_size, bid_price, bid_size) arrays of length n; empty levels are NaN price, 0 size."""
        out = []
        for side in (SELL, BUY):
            lv = self.levels(side, n)
            price = np.full(n, np.nan)
            size = np.zeros(n)
            for i, (p, s) in enumerate(lv):
                price[i], size[i] = p, s
            out += [price, size]
        return out[0], out[1], out[2], out[3]

    def volume_at(self, side: int, price: int) -> int:
        return (self.bid_volume if side == BUY else self.ask_volume).get(price, 0)

    # ------------------------------------------------------------ actions
    def _book(self, side: int):
        return (self.bids, self.bid_volume) if side == BUY else (self.asks, self.ask_volume)

    def _match(self, taker_side: int, size: int, owner: str, limit: int | None,
               self_trade_prevention: bool = False) -> tuple[list[Fill], int]:
        """Match against the opposite side up to `limit` price (None = any). Returns fills and unfilled size.

        With self_trade_prevention the taker's own resting orders are skipped (they keep their queue position),
        as exchange self-trade prevention would do; a level holding only the taker's own orders is passed over."""
        fills: list[Fill] = []
        book, volume = self._book(-taker_side)
        passed: set[int] = set()
        while size > 0:
            candidates = [p for p in volume if p not in passed]
            if not candidates:
                break
            best = min(candidates) if taker_side == BUY else max(candidates)
            if limit is not None and ((taker_side == BUY and best > limit) or (taker_side == SELL and best < limit)):
                break
            queue = book[best]
            i = 0
            while size > 0 and i < len(queue):
                maker = queue[i]
                if maker.size == 0:          # lazily cancelled
                    del queue[i]
                    continue
                if self_trade_prevention and maker.owner == owner:
                    i += 1
                    continue
                qty = min(size, maker.size)
                maker.size -= qty
                volume[best] -= qty
                size -= qty
                fills.append(Fill(best, qty, maker.owner, maker.order_id, owner, taker_side))
                if maker.size == 0:
                    del queue[i]
                    self.orders.pop(maker.order_id, None)
            if volume.get(best, 0) <= 0:
                volume.pop(best, None)
                book.pop(best, None)
            elif size > 0:
                passed.add(best)             # only the taker's own orders remain at this level
        return fills, size

    def limit(self, side: int, price: int, size: int, owner: str) -> tuple[int | None, list[Fill]]:
        """Place a limit order. A marketable part executes immediately; the rest rests.
        Returns (resting order id or None if fully filled, fills)."""
        if size <= 0:
            raise ValueError("size must be positive")
        fills, remaining = self._match(side, size, owner, limit=price)
        if remaining == 0:
            return None, fills
        oid = self._next_id
        self._next_id += 1
        order = Order(oid, side, price, remaining, owner)
        book, volume = self._book(side)
        book.setdefault(price, deque()).append(order)
        volume[price] = volume.get(price, 0) + remaining
        self.orders[oid] = order
        return oid, fills

    def market(self, side: int, size: int, owner: str, self_trade_prevention: bool = False) -> list[Fill]:
        """Market order: walks the opposite side; any size the book cannot absorb is dropped."""
        if size <= 0:
            raise ValueError("size must be positive")
        fills, _ = self._match(side, size, owner, limit=None, self_trade_prevention=self_trade_prevention)
        return fills

    def cancel(self, order_id: int) -> int:
        """Cancel a resting order. Returns the size that was still resting (0 if unknown or already filled)."""
        order = self.orders.pop(order_id, None)
        if order is None or order.size == 0:
            return 0
        _, volume = self._book(order.side)
        remaining = order.size
        volume[order.price] -= remaining
        order.size = 0
        if volume[order.price] <= 0:
            volume.pop(order.price, None)
            self._book(order.side)[0].pop(order.price, None)
        return remaining
