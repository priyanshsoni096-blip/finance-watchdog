"""Episode start windows must give the splits disjoint stretches of the trading day."""
import pytest

from env.lob_env import EnvConfig, LimitOrderBookEnv
from env.normalization import reference_stats
from watchdog.dataset import TIME_SPLIT, split_window


def test_split_windows_are_disjoint_and_ordered():
    w = {s: split_window(600_000, s) for s in TIME_SPLIT}
    assert w["train"]["start_max"] <= w["test"]["start_min"]
    assert w["test"]["start_max"] <= w["heldout"]["start_min"]
    assert w["train"]["start_min"] >= 30_000


def test_episode_starts_stay_inside_the_window(day):
    d = day("GOOG")
    win = split_window(len(d), "heldout")
    env = LimitOrderBookEnv(EnvConfig(ticker="GOOG", events_per_step=10, episode_len=50, **win),
                            day=d, stats=reference_stats(d))
    for seed in range(12):
        env.reset(seed=seed)
        assert win["start_min"] <= env.t <= win["start_max"]


def test_train_and_heldout_episodes_cannot_overlap(day):
    d = day("GOOG")
    span = 10 * 50
    starts = {}
    for split in ("train", "heldout"):
        win = split_window(len(d), split)
        env = LimitOrderBookEnv(EnvConfig(ticker="GOOG", events_per_step=10, episode_len=50, **win),
                                day=d, stats=reference_stats(d))
        starts[split] = [(env.reset(seed=s), env.t)[1] for s in range(10)]
    assert max(starts["train"]) + span <= min(starts["heldout"])


def test_window_too_small_is_rejected(day):
    d = day("GOOG")
    with pytest.raises(ValueError, match="no room for an episode"):
        LimitOrderBookEnv(EnvConfig(ticker="GOOG", events_per_step=10, episode_len=200,
                                    start_min=40_000, start_max=40_500), day=d, stats=reference_stats(d))
