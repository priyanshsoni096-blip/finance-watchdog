import numpy as np
import pytest

from env.normalization import reference_stats
from env.price_impact import PriceImpact, calibrate_lambda

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]


def test_zero_lambda_means_zero_shift():
    assert PriceImpact(0.0).shift(1e6, 100, 0.01) == 0.0
    assert PriceImpact(0.0, "sqrt").shift(-1e6, 100, 0.01) == 0.0


def test_linear_sign_and_proportionality():
    imp = PriceImpact(2.0)
    up = imp.shift(1000, 500, 0.01)
    assert up > 0
    assert imp.shift(-1000, 500, 0.01) == pytest.approx(-up)
    assert imp.shift(2000, 500, 0.01) == pytest.approx(2 * up)
    assert imp.shift(0, 500, 0.01) == 0.0


def test_sqrt_is_concave():
    imp = PriceImpact(1.0, "sqrt")
    assert imp.shift(4000, 1000, 0.01) == pytest.approx(2 * imp.shift(1000, 1000, 0.01))


@pytest.mark.parametrize("ticker", TICKERS)
def test_ofi_calibration_explains_real_price_moves(day, ticker):
    d = day(ticker)
    s = reference_stats(d)
    lam, t = calibrate_lambda(d, s)
    print(f"{ticker}: lambda={lam:.4f} R2={t['ofi_r2']:.2f} n={t['ofi_samples']}")
    assert lam > 0
    assert t["ofi_r2"] > 0.2, "order-flow imbalance should explain a real share of mid moves"


@pytest.mark.parametrize("ticker", ["MSFT", "INTC"])
def test_percentile_rule_is_implausible_on_thick_books(day, ticker):
    """Records the negative result: the rejected rule implies >20-spread moves for a 10x-depth spoof."""
    d = day(ticker)
    s = reference_stats(d)
    _, t = calibrate_lambda(d, s)
    assert t["percentile_rule_lambda"] * 10 > 20


@pytest.mark.parametrize("ticker", TICKERS)
def test_spoof_of_10x_touch_depth_moves_price(day, ticker):
    d = day(ticker)
    s = reference_stats(d)
    lam, t = calibrate_lambda(d, s)
    spoof = 10 * t["median_touch_depth"]
    on = PriceImpact(lam).shift(spoof, t["median_touch_depth"], t["median_ref_spread"])
    off = PriceImpact(0.0).shift(spoof, t["median_touch_depth"], t["median_ref_spread"])
    spreads = on / t["median_ref_spread"]
    print(f"{ticker}: 10x-depth spoof shifts mid by {on:.4f}$ = {spreads:.2f} spreads")
    assert on > 0 and off == 0.0
    assert 0.1 < spreads < 5.0, "implausible impact magnitude"
