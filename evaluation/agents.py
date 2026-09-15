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
    """A frozen trained Spoofer. deterministic=False samples from the learned action distribution, which is
    the policy PPO actually optimised; taking the most likely action can lose the learned behaviour."""

    def __init__(self, path: Path, deterministic: bool = True):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic

    def reset(self, rng):
        pass

    def act(self, obs, env, rng):
        return int(self.model.predict(obs, deterministic=self.deterministic)[0])


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
        # hold time is 1-300 market events whatever the decision granularity
        per_step = env.cfg.events_per_step
        if env.spoofs:
            self.hold -= 1
            return CANCEL if self.hold <= 0 else NOOP
        if rng.random() < 0.02 * per_step:
            self.hold = -(-int(rng.integers(1, 300)) // per_step)
            return SPOOF_BUY if rng.random() < 0.5 else SPOOF_SELL
        return NOOP


class LateBurstSpoof:
    """Pre-registered clean held-out attacker (defined and committed before any data or results existed).

    Spoof one side, wait 100-200 market events, trade 4-8 lots on the other side in a burst, cancel
    immediately, then unwind. Its timing differs from SCRIPTED-ATK (wait 30-80, 1-5 trades, hold 0-20) and from
    the trained Spoofers. Behaviour is identical on every stock, which is what the stock-shortcut test needs.
    """

    def reset(self, rng):
        self.plan = []

    def act(self, obs, env, rng):
        if self.plan:
            return self.plan.pop(0)
        if env.inventory != 0:
            return SELL if env.inventory > 0 else BUY
        per_step = env.cfg.events_per_step
        if rng.random() < 0.01 * per_step:
            side = 1 if rng.random() < 0.5 else -1
            trade = SELL if side > 0 else BUY
            wait = -(-int(rng.integers(100, 201)) // per_step)
            self.plan = [NOOP] * wait + [trade] * int(rng.integers(4, 9)) + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP


class ScriptedSpoof:
    """Spoof one side, wait 30-80 market events for the impact to build, trade 1-5 lots on the other
    side, hold 0-20 events, cancel, then unwind. A single large order per cycle, not Coscia-style
    layering (the env allows one spoof per side).

    The wait matters: on the v3 env, trading immediately after placing the spoof lost on every stock,
    while waiting ~50 events was profitable on MSFT/INTC in a scripted probe. Waits are given in
    market events and converted to agent steps, so the attacker behaves the same at any
    events_per_step.
    """

    def reset(self, rng):
        self.plan = []

    def act(self, obs, env, rng):
        if self.plan:
            return self.plan.pop(0)
        if env.inventory != 0:
            return SELL if env.inventory > 0 else BUY
        per_step = env.cfg.events_per_step
        if rng.random() < 0.02 * per_step:
            side = 1 if rng.random() < 0.5 else -1
            trade = SELL if side > 0 else BUY
            wait = -(-int(rng.integers(30, 81)) // per_step)
            hold = -(-int(rng.integers(0, 21)) // per_step)
            self.plan = [NOOP] * wait + [trade] * int(rng.integers(1, 6)) + [NOOP] * hold + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP
