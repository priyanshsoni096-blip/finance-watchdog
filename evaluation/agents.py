"""Policies that drive LimitOrderBookEnv in evaluation and Watchdog data generation.

ModelPolicy wraps a frozen trained Spoofer. The scripted agents are controls:
  HONEST        legitimate trader, occasional market orders, no large orders
  FLICKER       large orders placed and cancelled with no trading (no intent to profit)
  SCRIPTED-ATK  hand-scripted spoofer: spoof one side, trade the other while it rests, cancel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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


class DripSpoof:
    """Pre-registered clean held-out attacker DRIP-ATK (CP-7; defined and committed before any of its data existed).

    Spoof one side, then make 2-4 single-lot trades on the OTHER side, each after a fresh 30-90 market-event wait.
    In each wait it trades one lot on the SAME side as the spoof with probability 0.5 (camouflage: activity that is
    not evidence of manipulation). Hold 0-50 events after the last opposite trade, cancel, then unwind.
    Sparser and more spread out than SCRIPTED-ATK (one wait, then a burst of 1-5 trades) and LATEBURST-ATK (one long
    wait, then 4-8 trades); no source used in training trades on both sides of its own resting order.
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
            opposite, same = (SELL, BUY) if side > 0 else (BUY, SELL)
            plan = []
            for _ in range(int(rng.integers(2, 5))):
                gap = [NOOP] * -(-int(rng.integers(30, 91)) // per_step)
                if rng.random() < 0.5:
                    gap[int(rng.integers(len(gap)))] = same
                plan += gap + [opposite]
            self.plan = plan + [NOOP] * -(-int(rng.integers(0, 51)) // per_step) + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP


class SameSideTrader:
    """Legitimate control SAMESIDE for the pre-registered DRIP-ATK test (committed with it, before any data existed).

    Places a large visible order on one side and, while it rests, makes 1-5 single-lot trades on the SAME side
    (genuine interest in that direction), cancels after 100-300 market events, then unwinds. It is large, active and
    cancels, so it looks busy, but it never trades against its own resting order: legitimate under the label.
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
            hold = [NOOP] * -(-int(rng.integers(100, 301)) // per_step)
            k = min(int(rng.integers(1, 6)), len(hold))
            for i in rng.choice(len(hold), size=k, replace=False):
                hold[int(i)] = BUY if side > 0 else SELL
            self.plan = hold + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP


def _log_uniform_steps(rng, lo_events: int, hi_events: int, per_step: int) -> int:
    """A wait drawn log-uniformly between lo and hi market events, converted to agent steps (at least 1)."""
    events = float(np.exp(rng.uniform(np.log(lo_events), np.log(hi_events))))
    return max(1, -(-int(round(events)) // per_step))


class VariedSpoof:
    """TRAINING-ONLY scripted Spoofer family VARIED-SPOOF (CP-7 round 2). Never used as a held-out attacker.

    Spoof one side; then 1-6 single-lot trades on the OTHER side, each after a wait drawn log-uniformly from 1 to 400
    market events, so the incriminating act happens at order ages from one step to several hundred events. In each
    wait, with probability 0.2, one same-side trade. Hold a log-uniform 1-200 events, cancel, unwind. It exists
    because the round-1 Watchdog only flagged acts on young orders: almost every act in the RL Spoofers' data comes
    1-4 steps after placement (UPDATES.md U-022).
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
            opposite, same = (SELL, BUY) if side > 0 else (BUY, SELL)
            plan = []
            for _ in range(int(rng.integers(1, 7))):
                gap = [NOOP] * (_log_uniform_steps(rng, 1, 400, per_step) - 1)
                if gap and rng.random() < 0.2:
                    gap[int(rng.integers(len(gap)))] = same
                plan += gap + [opposite]
            self.plan = plan + [NOOP] * _log_uniform_steps(rng, 1, 200, per_step) + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP


class Accumulator:
    """TRAINING-ONLY legitimate control ACCUM (CP-7 round 2), the counterpart of VARIED-SPOOF.

    A large visible order on one side resting a log-uniform 50-1,000 market events, with 0-6 single-lot trades on the
    SAME side while it rests, then cancel and unwind. Same order sizes, ages and trade counts as VARIED-SPOOF, so
    the only difference between the two is the side of the trades relative to the resting order.
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
            hold = [NOOP] * _log_uniform_steps(rng, 50, 1000, per_step)
            k = min(int(rng.integers(0, 7)), len(hold))
            for i in rng.choice(len(hold), size=k, replace=False):
                hold[int(i)] = BUY if side > 0 else SELL
            self.plan = hold + [CANCEL]
            return SPOOF_BUY if side > 0 else SPOOF_SELL
        return NOOP
