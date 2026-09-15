import numpy as np
import pytest

from env.normalization import (AGENT_CLIP, PRICE_CLIP, encode_agent, encode_book,
                               reference_stats)


def _medians(day):
    s = reference_stats(day)
    f = encode_book(day, s)
    p1 = np.median(np.abs(np.concatenate([f[:, 0], f[:, 2]])))
    s1 = np.median(np.concatenate([f[:, 1], f[:, 3]]))
    return p1, s1, f


def test_raw_tick_offsets_are_not_comparable(day):
    """Regression guard for the design finding: tick offsets differ >10x between AAPL and INTC."""
    aapl, intc = day("AAPL"), day("INTC")
    aapl_ticks = np.nanmedian(aapl.spread) / 0.01
    intc_ticks = np.nanmedian(intc.spread) / 0.01
    print(f"median spread ticks AAPL={aapl_ticks:.1f} INTC={intc_ticks:.1f}")
    assert aapl_ticks / intc_ticks > 10


def test_normalized_features_comparable_across_scale_extremes(day):
    p_a, s_a, _ = _medians(day("AAPL"))
    p_i, s_i, _ = _medians(day("INTC"))
    print(f"|best price feature| AAPL={p_a:.3f} INTC={p_i:.3f}; best size feature AAPL={s_a:.3f} INTC={s_i:.3f}")
    assert max(p_a, p_i) / min(p_a, p_i) < 2.0
    assert max(s_a, s_i) / min(s_a, s_i) < 2.0


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"])
def test_encoding_finite_bounded_and_ordered(day, ticker):
    d = day(ticker)
    s = reference_stats(d)
    f = encode_book(d, s)
    assert f.shape == (len(d), 40)
    assert np.isfinite(f).all()
    assert np.abs(f[:, 0::2]).max() <= PRICE_CLIP
    # best ask above mid, best bid below mid wherever the touch exists
    ok = ~np.isnan(d.mid)
    assert (f[ok, 0] > 0).all() and (f[ok, 2] < 0).all()


def test_reference_stats_are_causal(day):
    """Changing the future must not change reference statistics at earlier indices."""
    d = day("GOOG")
    n = 20_000
    s_full = reference_stats(d)
    cut = d.truncate_levels(10)
    from env.lobster_data import LobsterDay
    short = LobsterDay(cut.ticker, 10, *(getattr(cut, k)[:n] for k in
                       ("time", "event_type", "order_id", "size", "price", "direction",
                        "ask_price", "ask_size", "bid_price", "bid_size")))
    s_short = reference_stats(short)
    np.testing.assert_allclose(s_short.ref_spread, s_full.ref_spread[:n])
    np.testing.assert_allclose(s_short.ask_ref_size, s_full.ask_ref_size[:n])


def test_single_row_encoding_matches_batch(day):
    d = day("INTC")
    s = reference_stats(d)
    batch = encode_book(d, s, slice(1000, 1003))
    single = encode_book(d, s, 1001)
    np.testing.assert_allclose(single, batch[1])


def test_agent_features_clipped():
    f = encode_agent(1e9, -1e12, 1e9, mid=27.5, lot=100, max_inventory=1000, touch_depth=500)
    assert f.shape == (3,)
    assert np.all(np.abs(f) <= AGENT_CLIP)
    g = encode_agent(500, 27.5 * 100, 1000, mid=27.5, lot=100, max_inventory=1000, touch_depth=500)
    np.testing.assert_allclose(g, [0.5, 1.0, 2.0])
