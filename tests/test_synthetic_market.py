import math

import numpy as np
import pytest

from synthetic.market import BUY, SELL, OrderBook


def _book():
    b = OrderBook()
    b.limit(BUY, 99, 10, "a")
    b.limit(BUY, 98, 20, "b")
    b.limit(SELL, 101, 10, "c")
    b.limit(SELL, 102, 30, "d")
    return b


def test_best_prices_mid_and_levels():
    b = _book()
    assert (b.best_bid(), b.best_ask(), b.mid()) == (99, 101, 100.0)
    assert b.levels(BUY, 5) == [(99, 10), (98, 20)]
    assert b.levels(SELL, 1) == [(101, 10)]


def test_price_time_priority_and_partial_fill():
    b = _book()
    b.limit(SELL, 101, 5, "late")          # queues behind c at 101
    fills = b.market(BUY, 12, "taker")
    assert [(f.maker_owner, f.size, f.price) for f in fills] == [("c", 10, 101), ("late", 2, 101)]
    assert b.volume_at(SELL, 101) == 3
    assert all(f.taker_owner == "taker" and f.taker_side == BUY for f in fills)


def test_market_order_walks_levels_and_drops_unfillable_size():
    b = _book()
    fills = b.market(SELL, 50, "taker")
    assert [(f.price, f.size) for f in fills] == [(99, 10), (98, 20)]
    assert b.best_bid() is None and b.mid() is None


def test_marketable_limit_executes_then_rests():
    b = _book()
    oid, fills = b.limit(BUY, 101, 15, "x")
    assert [(f.maker_owner, f.size) for f in fills] == [("c", 10)]
    assert oid is not None and b.best_bid() == 101 and b.volume_at(BUY, 101) == 5
    assert b.best_ask() == 102


def test_cancel_is_skipped_by_matching_and_updates_volume():
    b = _book()
    oid, _ = b.limit(SELL, 101, 7, "spoof")
    b.limit(SELL, 101, 4, "after")
    assert b.cancel(oid) == 7
    assert b.volume_at(SELL, 101) == 14
    assert b.cancel(oid) == 0              # second cancel is a no-op
    fills = b.market(BUY, 14, "taker")
    assert [f.maker_owner for f in fills] == ["c", "after"]
    assert b.best_ask() == 102


def test_cancel_last_order_removes_level():
    b = OrderBook()
    oid, _ = b.limit(BUY, 50, 3, "x")
    assert b.cancel(oid) == 3
    assert b.best_bid() is None and 50 not in b.bids


def test_snapshot_padding_matches_lobster_convention():
    b = _book()
    ask_p, ask_s, bid_p, bid_s = b.snapshot(3)
    np.testing.assert_array_equal(ask_s, [10, 30, 0])
    assert ask_p[0] == 101 and ask_p[1] == 102 and math.isnan(ask_p[2])
    np.testing.assert_array_equal(bid_s, [10, 20, 0])


def test_self_trade_prevention_skips_own_orders_and_keeps_their_place():
    b = _book()                              # asks: 101 c x10, 102 d x30
    own, _ = b.limit(SELL, 101, 5, "me")     # queues behind c at 101
    b.limit(SELL, 101, 3, "e")               # behind my order
    fills = b.market(BUY, 12, "me", self_trade_prevention=True)
    assert [(f.maker_owner, f.size, f.price) for f in fills] == [("c", 10, 101), ("e", 2, 101)]
    assert b.orders[own].size == 5 and b.volume_at(SELL, 101) == 6


def test_self_trade_prevention_passes_level_with_only_own_orders():
    b = OrderBook()
    b.limit(SELL, 101, 5, "me")
    b.limit(SELL, 102, 4, "x")
    fills = b.market(BUY, 3, "me", self_trade_prevention=True)
    assert [(f.maker_owner, f.price, f.size) for f in fills] == [("x", 102, 3)]
    assert b.volume_at(SELL, 101) == 5
    assert b.market(BUY, 10, "me", self_trade_prevention=True)[0].maker_owner == "x"
    assert b.market(BUY, 10, "me", self_trade_prevention=True) == []   # only own order left


def test_without_prevention_own_orders_trade():
    b = OrderBook()
    b.limit(SELL, 101, 5, "me")
    assert b.market(BUY, 5, "me")[0].maker_owner == "me"


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        OrderBook().market(BUY, 0, "x")
