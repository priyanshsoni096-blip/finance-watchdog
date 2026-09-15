import numpy as np

from synthetic.market import BUY, SELL
from synthetic.simulator import MarketParams, SyntheticMarket


def test_same_seed_reproduces_the_market():
    a, b = SyntheticMarket(seed=3), SyntheticMarket(seed=3)
    ma, sa = a.run(3_000)
    mb, sb = b.run(3_000)
    np.testing.assert_array_equal(ma, mb)
    np.testing.assert_array_equal(sa, sb)
    assert not np.array_equal(ma, SyntheticMarket(seed=4).run(3_000)[0])


def test_book_stays_two_sided_and_spread_positive():
    m = SyntheticMarket(seed=11)
    for _ in range(20_000):
        m.step()
        assert m.book.best_bid() is not None and m.book.best_ask() is not None
        assert m.book.best_ask() > m.book.best_bid()


def test_followers_trade_toward_the_heavy_side():
    params = MarketParams(p_follower=1.0, follower_p_react=1.0, follower_threshold=0.1, cancel_hazard=0.0)
    m = SyntheticMarket(params, seed=5)
    bb = m.book.best_bid()
    m.book.limit(BUY, bb, 50_000, "spoofer")          # very heavy bid side
    assert m.imbalance() > 0.1
    fills = m.step()
    assert fills and all(f.taker_owner == "follower" and f.taker_side == BUY for f in fills)

    m2 = SyntheticMarket(params, seed=5)
    ba = m2.book.best_ask()
    m2.book.limit(SELL, ba, 50_000, "spoofer")
    fills = m2.step()
    assert fills and all(f.taker_side == SELL for f in fills)


def test_followers_idle_when_book_is_balanced():
    params = MarketParams(p_follower=1.0, follower_p_react=1.0, follower_threshold=0.35, cancel_hazard=0.0)
    m = SyntheticMarket(params, seed=2)
    assert abs(m.imbalance()) < 0.35                   # symmetric initial book
    fills = m.step()
    # an idle follower is not an event: a noise trader acts instead, and no follower trades
    assert m.counts["follower_market"] == 0
    assert m.counts["limit"] + m.counts["market"] == 1
    assert all(f.taker_owner != "follower" for f in fills)


def test_step_survives_a_side_emptied_between_events():
    m = SyntheticMarket(seed=12)
    m.run(200)
    m.book.market(BUY, 10**9, "agent")          # an agent sweeps the whole ask side between events
    assert m.book.best_ask() is None
    for _ in range(50):
        m.step()
        assert m.book.best_bid() is not None and m.book.best_ask() is not None


def test_counts_track_event_types():
    m = SyntheticMarket(seed=8)
    m.run(5_000)
    c = m.counts
    assert c["limit"] + c["cancel"] + c["market"] + c["follower_market"] == 5_000
    assert c["cancel"] > 0 and c["limit"] > 0
