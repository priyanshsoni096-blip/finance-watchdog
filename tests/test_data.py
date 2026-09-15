import numpy as np
import pytest

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]


def test_first_aapl_rows_match_verified_values(day):
    d = day("AAPL")
    # First message row: 34200.004241176, 1, 16113575, 18, 5853300, 1
    assert d.time[0] == pytest.approx(34200.004241176)
    assert d.event_type[0] == 1
    assert d.order_id[0] == 16113575
    assert d.size[0] == 18
    assert d.price[0] == pytest.approx(585.33)
    assert d.direction[0] == 1
    # First orderbook row: ask 585.94 x 200, bid 585.33 x 18
    assert d.ask_price[0, 0] == pytest.approx(585.94)
    assert d.ask_size[0, 0] == 200
    assert d.bid_price[0, 0] == pytest.approx(585.33)
    assert d.bid_size[0, 0] == 18


@pytest.mark.parametrize("ticker", TICKERS)
def test_book_is_sane(day, ticker):
    d = day(ticker)
    assert len(d) > 10_000
    assert np.all(np.diff(d.time) >= 0), "timestamps must be non-decreasing"
    both = ~np.isnan(d.ask_price[:, 0]) & ~np.isnan(d.bid_price[:, 0])
    assert both.mean() > 0.99
    assert np.all(d.spread[both] > 0), "crossed or locked book at the touch"
    # asks ascend, bids descend across levels (ignoring empty levels)
    with np.errstate(invalid="ignore"):
        assert not np.any(np.diff(d.ask_price, axis=1) <= 0)
        assert not np.any(np.diff(d.bid_price, axis=1) >= 0)


@pytest.mark.parametrize("ticker", TICKERS)
def test_row_alignment_new_orders_at_touch(day, ticker):
    """A new limit order that becomes the best bid/ask must appear in the book row of the same index."""
    d = day(ticker)
    new_buy = (d.event_type == 1) & (d.direction == 1)
    at_best = new_buy & np.isclose(d.price, d.bid_price[:, 0])
    idx = np.flatnonzero(new_buy)
    # every new buy order priced strictly above the prior best bid must set the new best bid
    prev_best = np.concatenate([[np.nan], d.bid_price[:-1, 0]])
    improving = idx[(d.price[idx] > prev_best[idx]) & (d.price[idx] < d.ask_price[idx, 0])]
    assert len(improving) > 100
    assert np.all(at_best[improving])
