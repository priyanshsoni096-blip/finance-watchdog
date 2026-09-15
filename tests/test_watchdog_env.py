import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from evaluation.rollout import OBS_DIM
from watchdog.watchdog_env import WatchdogEnv, flag_episode


def _source(labels_per_episode, marker):
    y = np.concatenate([np.asarray(l, dtype=bool) for l in labels_per_episode])
    X = np.zeros((len(y), OBS_DIM), dtype=np.float32)
    X[:, 0] = np.arange(len(y))          # step index lets tests check replay order
    X[:, 1] = marker
    X[:, 2] = y                           # "cheating" feature for the dummy policy
    off = np.concatenate([[0], np.cumsum([len(l) for l in labels_per_episode])])
    return {"X": X, "y": y, "ep_offsets": off}


SOURCES = {"SPOOF_A": _source([[0, 1, 1, 0, 0], [1, 1, 0]], 1.0), "CLEAN_B": _source([[0, 0, 0, 0]], 2.0)}


def test_check_env():
    check_env(WatchdogEnv(SOURCES), skip_render_check=True)


def test_reward_table_and_replay_order():
    env = WatchdogEnv(SOURCES, tp_reward=1, fn_cost=1, fp_cost=2)
    obs, info = env.reset(options={"source": "SPOOF_A", "episode": 0})
    assert info["source"] == "SPOOF_A" and obs[0] == 0
    rewards, trunc = [], False
    for action in [1, 1, 0, 0, 1]:        # labels 0 1 1 0 0 -> FP, TP, FN, TN, FP
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        assert not term
    assert rewards == [-2, 1, -1, 0, -2]
    assert trunc


def test_second_episode_boundaries():
    env = WatchdogEnv(SOURCES)
    obs, _ = env.reset(options={"source": "SPOOF_A", "episode": 1})
    assert obs[0] == 5                     # episode 1 starts at global row 5
    steps = 0
    while True:
        *_, trunc, _ = env.step(0)
        steps += 1
        if trunc:
            break
    assert steps == 3


def test_sources_sampled_equally():
    env = WatchdogEnv(SOURCES)
    counts = {"SPOOF_A": 0, "CLEAN_B": 0}
    for s in range(400):
        _, info = env.reset(seed=s)
        counts[info["source"]] += 1
    assert 150 < counts["SPOOF_A"] < 250


class _Dummy:
    """Flags when feature 2 is set; checks the recurrent-state protocol is followed."""

    def __init__(self):
        self.starts = []

    def predict(self, x, state=None, episode_start=None, deterministic=True):
        self.starts.append(bool(episode_start[0]))
        return np.array([int(x[0, 2] > 0.5)]), ("h", "c")


class _Recorder:
    def __init__(self):
        self.seen = []

    def predict(self, x, state=None, episode_start=None, deterministic=True):
        self.seen.append(x.copy())
        return np.array([0]), None


def test_flag_episode_applies_obs_normalisation():
    m = _Recorder()
    X = np.full((2, OBS_DIM), 5.0, dtype=np.float32)
    X[1, 0] = 100.0                                   # env bound clips this to 25 before normalising
    norm = {"mean": np.ones(OBS_DIM, np.float32), "var": np.full(OBS_DIM, 4.0, np.float32), "epsilon": 0.0, "clip": 10.0}
    flag_episode(m, X, norm)
    assert m.seen[0][0, 1] == pytest.approx((5 - 1) / 2)
    assert m.seen[1][0, 0] == pytest.approx(10.0)     # (25 - 1) / 2 = 12, clipped to 10


def test_flag_episode_uses_state_protocol():
    m = _Dummy()
    X = SOURCES["SPOOF_A"]["X"][:5]
    flags = flag_episode(m, X)
    assert flags.tolist() == [False, True, True, False, False]
    assert m.starts == [True, False, False, False, False]
