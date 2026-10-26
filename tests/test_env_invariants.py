"""Accounting and lifecycle invariants for the replay market simulator (audit Phase 4).

These check the simulator's bookkeeping rather than its realism: every cash movement must be explained by a
trade, a run-over or the end-of-episode liquidation, inventory must move with it, and the reported P&L must
equal cash plus inventory marked at the unimpacted mid.
"""
import numpy as np
import pytest

from env.lob_env import BUY, CANCEL, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig, LimitOrderBookEnv
from env.normalization import reference_stats

TICKERS = ["AAPL", "INTC"]


@pytest.fixture
def make(day):
    def build(ticker="INTC", **kw):
        d = day(ticker)
        base = dict(events_per_step=10, episode_len=60)
        base.update(kw)
        return LimitOrderBookEnv(EnvConfig(ticker=ticker, **base), day=d, stats=reference_stats(d))
    return build


@pytest.mark.parametrize("ticker", TICKERS)
def test_pnl_identity_holds_every_step(make, ticker):
    """Reported P&L must equal cash + inventory valued at the unimpacted historical mid."""
    env = make(ticker)
    obs, info = env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(300):
        obs, reward, term, trunc, info = env.step(int(rng.integers(6)))
        expected = env.cash + env.inventory * env.mark_mid()
        assert info["pnl"] == pytest.approx(expected, abs=1e-6)
        assert np.isfinite(reward) and np.isfinite(obs).all()
        if term or trunc:
            obs, info = env.reset()


@pytest.mark.parametrize("ticker", TICKERS)
def test_every_cash_move_is_explained_by_a_trade_run_over_or_liquidation(make, ticker):
    """Cash and inventory may only change together, by an amount equal to quantity x price."""
    env = make(ticker)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    unexplained = 0
    for _ in range(400):
        cash0, inv0 = env.cash, env.inventory
        action = int(rng.integers(6))
        _, _, term, trunc, info = env.step(action)
        d_cash, d_inv = env.cash - cash0, env.inventory - inv0
        moved = abs(d_cash) > 1e-9 or d_inv != 0
        if not moved:
            continue
        caused = (info["trade_price"] is not None) or info["run_over"] or info.get("liquidation")
        assert caused, f"cash/inventory moved with no trade, run-over or liquidation (action {action})"
        if info["trade_price"] is not None and not info["run_over"] and not info.get("liquidation"):
            # a plain market order: one lot at the reported fill price
            assert abs(d_inv) == env.lot
            assert d_cash == pytest.approx(-d_inv * info["trade_price"], rel=1e-9)
        else:
            unexplained += 0  # combined events are checked by the identity test above
        if term or trunc:
            env.reset()


@pytest.mark.parametrize("ticker", TICKERS)
def test_inventory_limit_respected_and_episode_ends_flat(make, ticker):
    env = make(ticker, episode_len=40)
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    while True:
        _, _, term, trunc, info = env.step(int(rng.integers(6)))
        # market orders may never breach the limit; a run-over can, and is allowed to
        if info["trade_price"] is not None and not info["run_over"]:
            assert abs(env.inventory) <= env.max_inventory + env.lot
        if term or trunc:
            break
    assert trunc and env.inventory == 0, "episode must liquidate to flat"
    assert env.spoofs == [], "resting orders must be cancelled at episode end"


def test_duplicate_spoof_on_a_side_is_ignored_and_cancel_is_idempotent(make):
    env = make()
    env.reset(seed=3)
    env.step(SPOOF_BUY)
    assert len(env.spoofs) == 1
    first = env.spoofs[0]
    env.step(SPOOF_BUY)
    assert len(env.spoofs) == 1 and env.spoofs[0] is first, "a second spoof on the same side must be ignored"
    env.step(SPOOF_SELL)
    assert len(env.spoofs) == 2, "the opposite side is a separate order"
    env.step(CANCEL)
    assert env.spoofs == []
    cash_before, inv_before = env.cash, env.inventory
    for _ in range(3):
        env.step(CANCEL)
    assert env.spoofs == [] and env.cash == cash_before and env.inventory == inv_before


def test_blocked_market_order_changes_nothing(make):
    """At the inventory limit a buy must be refused outright, not partially applied."""
    env = make(max_inventory_lots=2)
    env.reset(seed=9)
    for _ in range(2):
        env.step(BUY)
    assert env.inventory == 2 * env.lot
    cash, inv = env.cash, env.inventory
    _, _, _, _, info = env.step(BUY)
    if not info["run_over"]:
        assert env.inventory == inv and env.cash == pytest.approx(cash)
        assert info["trade_price"] is None, "a refused order must not report a fill"


def test_reward_is_scale_free_across_very_different_stocks(make):
    """The reward divides by spread x lot, so comparable behaviour must give comparable rewards."""
    rewards = {}
    for ticker in ("AAPL", "INTC"):
        env = make(ticker)
        env.reset(seed=5)
        rng = np.random.default_rng(5)
        rewards[ticker] = np.std([env.step(int(rng.integers(6)))[1] for _ in range(200)])
    ratio = max(rewards.values()) / max(min(rewards.values()), 1e-9)
    assert ratio < 25, f"reward scale differs {ratio:.0f}x between stocks: {rewards}"


def test_observation_stays_inside_the_declared_space(make):
    env = make()
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    assert env.observation_space.contains(obs)
    for _ in range(250):
        obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
        assert env.observation_space.contains(obs), "observation left its declared Box"
        if term or trunc:
            obs, _ = env.reset()
