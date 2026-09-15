"""WatchdogEnv — the detector's RL environment.

The Watchdog watches one participant's recorded episode step by step (46-dim surveillance
observation, see evaluation/rollout.py) and chooses 0 = clear or 1 = flag.

Reward per step (asymmetric: a false accusation costs more than a miss, per the design):
    flag while a manipulative order rests      +tp_reward   (default +1)
    clear while a manipulative order rests     -fn_cost     (default -1)
    flag when nothing manipulative rests       -fp_cost     (default -2)
    clear when nothing manipulative rests       0

The Watchdog is never told the rule "large order + opposite trade + cancel"; it only receives these
rewards. Episodes are sampled source-first (each SOURCE_TICKER equally likely), so sources that rarely
spoof are not drowned out by long clean episodes.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evaluation.rollout import OBS_DIM

OBS_BOUND = 25.0


class WatchdogEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sources: dict[str, dict], tp_reward: float = 1.0, fn_cost: float = 1.0,
                 fp_cost: float = 2.0):
        super().__init__()
        if not sources:
            raise ValueError("no sources given")
        self.tp_reward, self.fn_cost, self.fp_cost = tp_reward, fn_cost, fp_cost
        self.names = sorted(sources)
        self.episodes = {}
        for name in self.names:
            src = sources[name]
            off = src["ep_offsets"]
            self.episodes[name] = [(src["X"][a:b], src["y"][a:b]) for a, b in zip(off[:-1], off[1:]) if b > a]
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _obs(self) -> np.ndarray:
        return np.clip(self.X[self.i], -OBS_BOUND, OBS_BOUND).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.source = options.get("source", self.names[int(self.np_random.integers(len(self.names)))])
        eps = self.episodes[self.source]
        k = int(options.get("episode", self.np_random.integers(len(eps))))
        self.X, self.y = eps[k]
        self.i = 0
        return self._obs(), {"source": self.source}

    def step(self, action):
        flag, label = int(action) == 1, bool(self.y[self.i])
        if label:
            reward = self.tp_reward if flag else -self.fn_cost
        else:
            reward = -self.fp_cost if flag else 0.0
        info = {"label": label, "flag": flag, "source": self.source}
        self.i += 1
        truncated = self.i >= len(self.y)
        if truncated:
            self.i = len(self.y) - 1
        return self._obs(), float(reward), False, truncated, info


def flag_episode(model, X: np.ndarray) -> np.ndarray:
    """Run a (recurrent) policy deterministically over one recorded episode; returns per-step flags."""
    X = np.clip(X, -OBS_BOUND, OBS_BOUND).astype(np.float32)
    state, start = None, np.ones((1,), dtype=bool)
    flags = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        action, state = model.predict(X[i][None], state=state, episode_start=start, deterministic=True)
        start = np.zeros((1,), dtype=bool)
        flags[i] = int(np.asarray(action).reshape(-1)[0]) == 1
    return flags
