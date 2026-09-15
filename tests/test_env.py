from functools import lru_cache

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.lob_env import (BUY, CANCEL, NOOP, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig,
                         LimitOrderBookEnv)
from env.normalization import reference_stats

TICKERS = ["AAPL", "MSFT", "GOOG", "INTC", "AMZN"]


@lru_cache(maxsize=None)
def _stats(ticker):
    from tests.conftest import _day
    return reference_stats(_day(ticker))


@pytest.fixture
def make(day):
    def build(ticker="INTC", **kw):
        d = day(ticker)
        return LimitOrderBookEnv(EnvConfig(ticker=ticker, **kw), day=d, stats=_stats(ticker))
    return build


def _quiet_start(env, horizon=60):
    """A start index where no resting spoof at the touch would be run over within `horizon` events."""
    d = env.day
    for t in range(env.cfg.warmup, env._last_start, 997):
        w = slice(t + 1, t + horizon + 1)
        if (np.nanmin(d.ask_price[w, 0]) > d.bid_price[t, 0]
                and np.nanmax(d.bid_price[w, 0]) < d.ask_price[t, 0]
                and np.isfinite(d.mid[t:t + horizon + 1]).all()):
            return t
    raise AssertionError("no quiet window found")


def _run(env, start, actions):
    env.reset(options={"start": start})
    total, infos = 0.0, []
    for a in actions:
        _, r, term, trunc, info = env.step(a)
        total += r
        infos.append(info)
        assert not term
    return total, infos


def test_check_env(make):
    check_env(make("GOOG", episode_len=50), skip_render_check=True)


def test_spaces(make):
    env = make("AAPL")
    assert env.observation_space.shape == (43,)
    assert env.action_space.n == 6


def test_spoof_visible_in_observation(make):
    env = make("INTC")
    start = _quiet_start(env)
    obs0, _ = env.reset(options={"start": start})
    obs1, *_ = env.step(SPOOF_BUY)
    env2 = make("INTC")
    env2.reset(options={"start": start})
    obs_noop, *_ = env2.step(NOOP)
    # best bid size feature (index 3) must be larger with the spoof resting
    assert obs1[3] > obs_noop[3]
    assert obs1[42] > 0 and obs_noop[42] == 0


@pytest.mark.parametrize("ticker", ["AAPL", "INTC"])
def test_holding_through_spoof_earns_nothing_extra(make, ticker):
    """Reward-hack guard: long + spoof buy + hold + cancel, no trade at the distorted price,
    must produce exactly the same reward stream as the same sequence without the spoof."""
    a, b = make(ticker), make(ticker)
    start = _quiet_start(a)
    with_spoof = [BUY, SPOOF_BUY] + [NOOP] * 30 + [CANCEL, NOOP]
    without = [BUY, NOOP] + [NOOP] * 30 + [NOOP, NOOP]
    r_a, i_a = _run(a, start, with_spoof)
    r_b, _ = _run(b, start, without)
    assert max(abs(i["impact_shift"]) for i in i_a) > 0, "impact never engaged"
    assert all(not i["run_over"] for i in i_a)
    assert r_a == pytest.approx(r_b, abs=1e-9)


@pytest.mark.parametrize("ticker", ["AAPL", "INTC"])
def test_spoof_then_trade_is_profitable_only_with_impact(make, ticker):
    on, off = make(ticker), make(ticker, impact_lambda=0.0)
    hold = 250  # let the spoof's impact build up fully (ramp reaches 1 at 200 events)
    start = _quiet_start(on, horizon=hold + 10)
    actions = [SPOOF_BUY] + [NOOP] * hold + [SELL, CANCEL, NOOP, BUY, NOOP]
    _, i_on = _run(on, start, actions)
    _, i_off = _run(off, start, actions)
    gain = i_on[-1]["pnl"] - i_off[-1]["pnl"]
    shift_at_sell = i_on[hold]["impact_shift"]  # shift in force at the step where SELL executes
    print(f"{ticker}: extra PnL from spoof-then-sell = ${gain:.2f}; shift at sell = {shift_at_sell:.4f}")
    assert shift_at_sell > 0
    # self-impact of the sell and the later buy-back cancel up to intraday depth/spread drift
    assert gain == pytest.approx(on.lot * shift_at_sell, rel=0.05)
    assert gain > 0


def test_self_impact_closes_unlimited_buying_under_spoof(make):
    """Regression for the exploit SPOOFER-04 found: spoof sell, then buy repeatedly at the depressed
    price. Once bought volume reaches the spoof size, self-impact cancels the spoof shift, so every
    further buy fills at or above the historical ask."""
    env = make("INTC")
    env.reset(options={"start": _quiet_start(env)})
    env.step(SPOOF_SELL)
    spoof = env.spoofs[0].size
    checked = 0
    for _ in range(env.cfg.max_inventory_lots):
        t = env.t
        *_, info = env.step(BUY)
        if info["trade_price"] is None or not env.spoofs:
            break
        if env.volume > spoof:  # this buy started at or beyond the spoof size
            assert info["trade_price"] >= env.day.ask_price[t, 0] - 1e-9
            checked += 1
    assert checked > 0


def test_spoof_impact_builds_with_age_and_resets_on_replace(make):
    env = make("INTC")
    env.reset(options={"start": _quiet_start(env, horizon=260)})
    env.step(SPOOF_SELL)
    young = env.impact_shift()                      # age 1 event
    for _ in range(199):
        env.step(NOOP)
    t = env.t
    full = env.impact.shift(-env.spoofs[0].size, env.stats.touch_depth[t], env.stats.ref_spread[t])
    print(f"INTC spoof shift at age 1: {young:+.5f}, age 200: {env.impact_shift():+.5f}, full: {full:+.5f}")
    assert abs(young) < 0.2 * abs(full)
    assert env.impact_shift() == pytest.approx(full, rel=1e-9)
    env.step(CANCEL)
    env.step(SPOOF_SELL)
    assert abs(env.impact_shift()) < 0.2 * abs(full)


def test_spoof_benefit_capped_at_spoof_size(make):
    """After buying as many shares as the spoof's size under it, further buys get no spoof discount."""
    env = make("INTC", max_inventory_lots=100)
    env.reset(options={"start": _quiet_start(env, horizon=320)})
    env.step(SPOOF_SELL)
    for _ in range(200):
        env.step(NOOP)
    spoof = env.spoofs[0]
    while spoof.capacity > 0:
        env.step(BUY)
    assert env.spoofs and env.spoofs[0] is spoof
    t, v = env.t, env.volume
    *_, info = env.step(BUY)
    expected = env.day.ask_price[t, 0] + env.self_shift(v + env.lot / 2, t)
    assert info["trade_price"] == pytest.approx(expected, abs=1e-9)


def test_one_sided_trading_without_spoof_loses(make):
    env = make("MSFT")
    env.reset(options={"start": _quiet_start(env)})
    for _ in range(20):
        env.step(BUY)
    for _ in range(20):
        env.step(SELL)
    assert env.pnl() < 0


def test_run_over_fills_resting_spoof(make):
    env = make("INTC")
    d = env.day
    # find a real moment where the best ask later falls to the current best bid
    for t in range(env.cfg.warmup, env._last_start, 53):
        w = d.ask_price[t + 1:t + 400, 0]
        hit = np.flatnonzero(w <= d.bid_price[t, 0])
        if hit.size and np.isfinite(d.mid[t:t + hit[0] + 2]).all():
            break
    else:
        pytest.fail("no ask-through-bid move found")
    env.reset(options={"start": t})
    env.step(SPOOF_BUY)
    size = env.spoofs[0].size
    filled = None
    for _ in range(hit[0] + 1):
        *_, info = env.step(NOOP)
        if info["run_over"]:
            filled = info
            break
    assert filled is not None
    assert env.inventory == size
    assert env.spoofs == []


def test_bid_only_constraint(make):
    env = make("GOOG", bid_only=True)
    env.reset(options={"start": _quiet_start(env)})
    env.step(SPOOF_SELL)
    assert env.spoofs == []
    env.step(SPOOF_BUY)
    assert len(env.spoofs) == 1


def test_inventory_limit_on_market_orders(make):
    env = make("MSFT", max_inventory_lots=2)
    env.reset(options={"start": _quiet_start(env)})
    for _ in range(5):
        env.step(BUY)
    assert env.inventory == 2 * env.lot


@pytest.mark.parametrize("ticker", TICKERS)
def test_spoof_to_lot_ratio_comparable_across_tickers(make, ticker):
    """A spoof must be a similar number of lots on every stock, and fit inside the inventory limit."""
    env = make(ticker)
    env.reset(options={"start": _quiet_start(env)})
    env.step(SPOOF_BUY)
    lots = env.spoofs[0].size / env.lot
    print(f"{ticker}: lot={env.lot} spoof={env.spoofs[0].size:.0f} shares = {lots:.1f} lots")
    assert 5 <= lots <= 40
    assert env.spoofs[0].size <= env.max_inventory


def test_run_over_does_not_terminate_and_liquidation_crosses_spread(make):
    env = make("INTC", episode_len=400)
    d = env.day
    for t in range(env.cfg.warmup, env._last_start, 53):
        hit = np.flatnonzero(d.ask_price[t + 1:t + 300, 0] <= d.bid_price[t, 0])
        if hit.size and np.isfinite(d.mid[t:t + 402]).all():
            break
    env.reset(options={"start": t})
    env.step(SPOOF_BUY)
    saw_fill = False
    for _ in range(399):  # 1 spoof step + 399 no-ops = episode_len
        *_, term, trunc, info = env.step(NOOP)
        assert not term
        saw_fill |= bool(info["run_over"])
        if trunc:
            break
    assert saw_fill
    assert trunc
    liq = info["liquidation"]
    assert liq["qty"] > 0 and env.inventory == 0
    # long position sold at the bid, i.e. below the mid it was marked at
    assert liq["price"] < d.mid[env.t]


@pytest.mark.parametrize("ticker", TICKERS)
def test_random_rollout_finite(make, ticker):
    env = make(ticker)
    obs, _ = env.reset(seed=0)
    rewards = []
    for _ in range(10_000):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        assert env.observation_space.contains(obs)
        rewards.append(r)
        if term or trunc:
            obs, _ = env.reset()
    print(f"{ticker}: random-policy reward mean={np.mean(rewards):.4f} p1={np.percentile(rewards, 1):.3f} "
          f"p99={np.percentile(rewards, 99):.3f}")
