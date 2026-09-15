import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from env.lob_env import BUY, CANCEL, NOOP, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig
from env.lobster_data import LobsterDay
from env.normalization import ReferenceStats, encode_book
from synthetic.market import BUY as MBUY
from synthetic.market import SELL as MSELL
from synthetic.synthetic_env import AGENT, LiveSpoof, OnlineReferenceStats, SyntheticSpoofEnv, encode_live_book


def _env(**cfg):
    base = dict(events_per_step=10, episode_len=30)
    base.update(cfg)
    return SyntheticSpoofEnv(EnvConfig(**base), warmup_events=2_000)


def test_live_encoding_matches_replay_encoding():
    rng = np.random.default_rng(0)
    L = 10
    ask_p = 100 + np.arange(1, L + 1, dtype=float)
    bid_p = 100 - np.arange(0, L, dtype=float)
    ask_p[-1] = np.nan
    ask_s, bid_s = rng.integers(1, 5000, L).astype(float), rng.integers(1, 5000, L).astype(float)
    ask_s[-1] = 0.0
    ask_ref, bid_ref = rng.uniform(100, 3000, L), rng.uniform(100, 3000, L)
    day = LobsterDay("SYN", L, np.zeros(1), np.ones(1, np.int8), np.zeros(1, np.int64), np.ones(1, np.int64),
                     np.ones(1), np.ones(1, np.int8), ask_p[None], ask_s[None], bid_p[None], bid_s[None])
    stats = ReferenceStats(np.array([1.5]), ask_ref[None], bid_ref[None])
    np.testing.assert_allclose(encode_live_book(ask_p, ask_s, bid_p, bid_s, ask_ref, bid_ref, 1.5),
                               encode_book(day, stats, 0), rtol=1e-6)


def test_online_stats_match_pandas_reference():
    rng = np.random.default_rng(1)
    n, L = 400, 3
    ask = rng.integers(0, 400, (n, L)).astype(float)
    bid = rng.integers(0, 400, (n, L)).astype(float)
    spreads = rng.integers(1, 4, n).astype(float)
    online = OnlineReferenceStats(L, spread_window=50, size_halflife=30)
    for i in range(n):
        online.update(ask[i], bid[i], spreads[i])
    ewm = lambda x: pd.DataFrame(np.where(x > 0, x, np.nan)).ewm(halflife=30, adjust=False, ignore_na=True).mean() \
        .ffill().fillna(1.0).clip(lower=1.0).to_numpy()  # noqa: E731
    expected_depth = (ewm(ask)[:, 0] + ewm(bid)[:, 0]) / 2
    np.testing.assert_allclose(online.touch_depth, expected_depth, rtol=1e-9)
    expected_spread = pd.Series(spreads).rolling(50, min_periods=1).median().to_numpy()
    np.testing.assert_allclose(online.ref_spread, expected_spread)


def test_check_env():
    check_env(_env(), skip_render_check=True)


def test_spoof_rests_in_book_and_cancel_removes_it():
    env = _env()
    env.reset(seed=3)
    bb = env.market.book.best_bid()
    env.step(SPOOF_BUY)
    assert len(env.spoofs) == 1
    spoof = env.spoofs[0]
    assert spoof.order_id in env.market.book.orders and env.market.book.orders[spoof.order_id].owner == AGENT
    assert spoof.price == pytest.approx(bb * env.tick)
    env.step(CANCEL)
    assert env.spoofs == [] and spoof.order_id not in env.market.book.orders


def test_market_buy_fills_and_self_trade_prevention_protects_own_spoof():
    env = _env()
    env.reset(seed=5)
    env.step(SPOOF_SELL)
    spoof = env.spoofs[0]
    size_before = env.market.book.orders[spoof.order_id].size
    best_ask_before = env.market.book.best_ask()
    *_, info = env.step(BUY)
    assert info["trade_price"] is not None and info["trade_price"] >= best_ask_before * env.tick - 1e-9
    assert env.inventory == env.lot
    if spoof in env.spoofs:                          # not run over by background flow meanwhile
        assert env.market.book.orders[spoof.order_id].size <= size_before


def test_run_over_books_inventory_and_reports():
    env = _env()
    env.reset(seed=7)
    env.step(SPOOF_BUY)
    spoof = env.spoofs[0]
    size = int(env.market.book.orders[spoof.order_id].size)
    fills = env.market.book.market(MSELL, 10**9, "noise")        # sweep the whole bid side
    run_over = env._book_agent_fills(fills, env.t)
    assert env.inventory == size and env.spoofs == []
    assert run_over and run_over[0]["side"] == 1 and run_over[0]["size"] == size


def test_liquidation_flattens_at_episode_end():
    env = _env(episode_len=3)
    env.reset(seed=9)
    env.step(BUY)
    env.step(NOOP)
    *_, trunc, info = env.step(NOOP)
    assert trunc and info["liquidation"]["qty"] == env.lot and env.inventory == 0


def _imbalance_without_agent(env, levels):
    bid = sum(s - env._agent_volume(1, p) for p, s in env.market.book.levels(MBUY, levels))
    ask = sum(s - env._agent_volume(-1, p) for p, s in env.market.book.levels(MSELL, levels))
    return (bid - ask) / (bid + ask)


def test_blind_followers_ignore_agent_volume():
    cfg = EnvConfig(events_per_step=10, episode_len=30)
    blind = SyntheticSpoofEnv(cfg, warmup_events=500, blind_followers=True)
    seeing = SyntheticSpoofEnv(cfg, warmup_events=500, blind_followers=False)
    blind.reset(seed=11)
    seeing.reset(seed=11)
    for env in (blind, seeing):
        env.market.book.limit(MBUY, env.market.book.best_bid(), 50_000, AGENT)
        oid = max(env.market.book.orders)
        price = env.market.book.best_bid() * env.tick
        env.spoofs.append(LiveSpoof(oid, 1, 50_000.0, price, env.t, 50_000.0))
    n = blind.market.p.follower_levels
    assert blind.market.imbalance() == pytest.approx(_imbalance_without_agent(blind, n))
    assert seeing.market.imbalance() > blind.market.imbalance() + 0.1   # the visible spoof makes bids look heavier
