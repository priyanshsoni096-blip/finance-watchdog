"""Policies that drive LimitOrderBookEnv in evaluation and Watchdog data generation.

ModelPolicy wraps a frozen trained Spoofer. The scripted agents are controls:
  HONEST        legitimate trader, occasional market orders, no large orders
  FLICKER       large orders placed and cancelled with no trading (no intent to profit)
  SCRIPTED-ATK  hand-scripted spoofer: spoof one side, trade the other while it rests, cancel
"""
from __future__ import annotations

from pathlib import Path

from env.lob_env import BUY, CANCEL, NOOP, SELL, SPOOF_BUY, SPOOF_SELL


class ModelPolicy:
    def __init__(self, path: Path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self, rng):
        pass

    def act(self, obs, env, rng):
        return int(self.model.predict(obs, deterministic=True)[0])


class Honest:
    def reset(self, rng):
        pass

    def act(self, obs, env, rng):
        if rng.random() > 0.1:
            return NOOP
        flatten = SELL if env.inventory > 0 else BUY
        other = BUY if flatten == SELL else SELL
        return flatten if rng.random() < 0.7 else other


class Flicker:
    def reset(self, rng):
        self.hold = 0

    def act(self, obs, env, rng):
        if env.spoofs:
            self.hold -= 1
            return CANCEL if self.hold <= 0 else NOOP
        if rng.random() < 0.02:
            self.hold = int(rng.integers(1, 300))
            return SPOOF_BUY if rng.random() < 0.5 else SPOOF_SELL
        return NOOP


class ScriptedSpoof:
    """A single large order per cycle, not Coscia-style layering (the env allows one spoof per side)."""

    def reset(self, rng):
        self.plan = []

    def act(self, obs, env, rng):
        if self.plan:
            return self.plan.pop(0)
        if env.inventory != 0:
            return SELL if env.inventory > 0 else BUY
        if rng.random() < 0.02:
            side = 1 if rng.random() < 0.5 else -1
            trade = SELL if side > 0 else BUY
            self.plan = [trade] * int(rng.integers(1, 6)) + [NOOP] * int(rng.integers(0, 50)) + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP
