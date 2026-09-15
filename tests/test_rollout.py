import numpy as np
import pytest

from env.lob_env import EnvConfig, LimitOrderBookEnv
from env.normalization import reference_stats
from evaluation.agents import Flicker, Honest, ScriptedSpoof
from evaluation.rollout import BOOK_DIM, OBS_DIM, record_episode
from watchdog.dataset import _orders_to_array, orders_from_array


def _env(day, ticker="GOOG", episode_len=600):
    d = day(ticker)
    return LimitOrderBookEnv(EnvConfig(ticker=ticker, episode_len=episode_len), day=d, stats=reference_stats(d))


def test_scripted_spoofer_labels_follow_manipulative_orders(day):
    env = _env(day)
    ep = record_episode(env, ScriptedSpoof(), np.random.default_rng(3), seed=11)
    assert ep["X"].shape == (ep["steps"], OBS_DIM)
    assert np.isfinite(ep["X"]).all()
    manip = [o for o in ep["orders"] if o.manipulative]
    assert manip, "scripted spoofer should produce at least one manipulative order in 600 steps"
    for o in manip:
        assert ep["y"][o.placed_step:o.removed_step].all()
    # trade direction feature is +-1 exactly on steps with a trade
    assert set(np.unique(ep["X"][:, BOOK_DIM + 4])) <= {-1.0, 0.0, 1.0}


def test_flicker_and_honest_have_no_positive_labels(day):
    env = _env(day)
    for policy in (Flicker(), Honest()):
        ep = record_episode(env, policy, np.random.default_rng(5), seed=21)
        assert not ep["y"].any()
    assert ep["manip_trades"] == 0


def test_resting_order_visible_in_participant_features(day):
    env = _env(day)
    ep = record_episode(env, Flicker(), np.random.default_rng(7), seed=31)
    resting = ep["X"][:, BOOK_DIM] > 0
    assert resting.any() and not resting.all()


def test_orders_record_events_per_step(day):
    d = day("GOOG")
    env = LimitOrderBookEnv(EnvConfig(ticker="GOOG", events_per_step=10, episode_len=100), day=d,
                            stats=reference_stats(d))
    ep = record_episode(env, Flicker(), np.random.default_rng(9), seed=51)
    assert ep["orders"] and all(o.events_per_step == 10 for o in ep["orders"])
    back = orders_from_array(_orders_to_array([ep["orders"]]))
    assert all(o.events_per_step == 10 for _, o in back)


def test_orders_roundtrip_through_array(day):
    env = _env(day)
    eps = [record_episode(env, ScriptedSpoof(), np.random.default_rng(i), seed=40 + i) for i in range(2)]
    back = orders_from_array(_orders_to_array([e["orders"] for e in eps]))
    flat = [(i, o) for i, e in enumerate(eps) for o in e["orders"]]
    assert len(back) == len(flat)
    for (ea, a), (eb, b) in zip(back, flat):
        assert ea == eb and a.order_id == b.order_id and a.removed_by == b.removed_by
        assert a.manipulative == b.manipulative and a.depth_mult == pytest.approx(b.depth_mult)
