"""LimitOrderBookEnv — historical LOBSTER replay with an injected trading agent.

Actions (Discrete(6)):
  0 no-op | 1 market buy 1 lot | 2 market sell 1 lot
  3 spoof buy  (K x touch depth) at best bid | 4 spoof sell (K x touch depth) at best ask
  5 cancel all resting spoofs

Observation (Box(43), float32): 40 normalized book features (spread units / log size ratio,
resting spoofs added to the displayed size at their price level) + 3 agent features.

Economics:
  * Market orders fill at the IMPACTED touch: historical ask/bid + impact shift from resting spoofs.
  * Inventory is marked to the UNIMPACTED historical mid. Holding a position while a spoof rests
    therefore earns nothing; profit requires actually trading at the distorted price.
  * A resting spoof buy is run over (filled at its price) when the historical best ask falls to
    or below it; a spoof sell when the best bid rises to or above it.
  * Reward = PnL delta in units of (trailing spread x lot) - inv_penalty * |inventory| / lot.
    Both terms are scale-free, so rewards are comparable across tickers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.lobster_data import LobsterDay, load_day
from env.normalization import (AGENT_CLIP, PRICE_CLIP, SIZE_CLIP, ReferenceStats,
                               encode_agent, encode_book, reference_stats)
from env.price_impact import PriceImpact, calibrate_lambda

NOOP, BUY, SELL, SPOOF_BUY, SPOOF_SELL, CANCEL = range(6)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "configs" / "calibration.json"


@dataclass
class EnvConfig:
    ticker: str = "AAPL"
    levels: int = 10
    lot: int = 100
    spoof_k: float = 10.0              # spoof size as multiple of trailing touch depth
    max_inventory_lots: int = 10
    inv_penalty: float = 0.001         # per lot held, per step, in reward units
    episode_len: int = 2_000           # agent steps
    events_per_step: int = 1
    warmup: int = 5_000                # events skipped so reference stats are settled
    impact_form: str = "linear"
    impact_lambda: float | None = None  # None -> per-ticker calibrated value
    bid_only: bool = False             # SPOOFER-03 constraint: spoof-sell disabled
    extra: dict = field(default_factory=dict)


@dataclass
class Spoof:
    side: int      # +1 buy (rests on bid), -1 sell (rests on ask)
    size: float
    price: float


class LimitOrderBookEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None, *, day: LobsterDay | None = None,
                 stats: ReferenceStats | None = None):
        super().__init__()
        cfg = config if isinstance(config, EnvConfig) else EnvConfig(**(config or {}))
        self.cfg = cfg
        self.day = day if day is not None else load_day(cfg.ticker, cfg.levels)
        self.stats = stats if stats is not None else reference_stats(self.day)
        lam = cfg.impact_lambda if cfg.impact_lambda is not None else self._calibrated_lambda()
        self.impact = PriceImpact(lam, cfg.impact_form)
        self.max_inventory = cfg.max_inventory_lots * cfg.lot

        n_book = 4 * cfg.levels
        low = np.concatenate([np.tile([-PRICE_CLIP, 0.0], 2 * cfg.levels), np.full(3, -AGENT_CLIP)])
        high = np.concatenate([np.tile([PRICE_CLIP, SIZE_CLIP], 2 * cfg.levels), np.full(3, AGENT_CLIP)])
        assert low.shape == (n_book + 3,)
        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        valid = np.isfinite(self.day.mid)
        self._valid = valid
        self._last_start = len(self.day) - cfg.episode_len * cfg.events_per_step - 1
        if self._last_start <= cfg.warmup:
            raise ValueError("episode longer than available data")

    # ------------------------------------------------------------------ setup
    def _calibrated_lambda(self) -> float:
        key = (self.cfg.ticker, self.cfg.impact_form)
        if CALIBRATION_PATH.exists():
            cal = json.loads(CALIBRATION_PATH.read_text())
            if cal.get("form") == self.cfg.impact_form and self.cfg.ticker in cal["tickers"]:
                return float(cal["tickers"][self.cfg.ticker]["lambda"])
        lam, _ = calibrate_lambda(self.day, self.stats, key[1])
        return lam

    # --------------------------------------------------------------- helpers
    def _net_resting(self) -> float:
        return sum(s.side * s.size for s in self.spoofs)

    def impact_shift(self, t: int | None = None) -> float:
        t = self.t if t is None else t
        return self.impact.shift(self._net_resting(), self.stats.touch_depth[t], self.stats.ref_spread[t])

    def mark_mid(self, t: int | None = None) -> float:
        """Unimpacted historical mid used to value inventory."""
        t = self.t if t is None else t
        return float(self.day.mid[t])

    def pnl(self) -> float:
        return self.cash + self.inventory * self.mark_mid()

    @property
    def spoofing_active(self) -> bool:
        return len(self.spoofs) > 0

    def _obs(self) -> np.ndarray:
        t, d = self.t, self.day
        ask_s, bid_s = d.ask_size[t].copy(), d.bid_size[t].copy()
        for s in self.spoofs:
            prices, sizes = (d.bid_price[t], bid_s) if s.side > 0 else (d.ask_price[t], ask_s)
            hit = np.flatnonzero(np.isclose(prices, s.price))
            if hit.size:
                sizes[hit[0]] += s.size
        book = encode_book(d, self.stats, t, ask_size=ask_s, bid_size=bid_s)
        agent = encode_agent(self.inventory, self.pnl(), sum(s.size for s in self.spoofs),
                             mid=self.mark_mid(), lot=self.cfg.lot, max_inventory=self.max_inventory,
                             touch_depth=self.stats.touch_depth[t])
        return np.concatenate([book, agent]).astype(np.float32)

    def _info(self, fills=()) -> dict:
        return {
            "t": self.t,
            "pnl": self.pnl(),
            "inventory": self.inventory,
            "cash": self.cash,
            "spoofing_active": self.spoofing_active,
            "impact_shift": self.impact_shift(),
            "run_over": list(fills),
        }

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start" in options:
            start = int(options["start"])
        else:
            start = int(self.np_random.integers(self.cfg.warmup, self._last_start))
        while not self._valid[start]:
            start += 1
        self.t = start
        self.steps = 0
        self.inventory = 0
        self.cash = 0.0
        self.spoofs: list[Spoof] = []
        self._prev_pnl = 0.0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        cfg, d, t = self.cfg, self.day, self.t
        shift = self.impact_shift(t)

        if action == BUY and self.inventory + cfg.lot <= self.max_inventory:
            self.inventory += cfg.lot
            self.cash -= cfg.lot * (d.ask_price[t, 0] + shift)
        elif action == SELL and self.inventory - cfg.lot >= -self.max_inventory:
            self.inventory -= cfg.lot
            self.cash += cfg.lot * (d.bid_price[t, 0] + shift)
        elif action in (SPOOF_BUY, SPOOF_SELL):
            side = 1 if action == SPOOF_BUY else -1
            allowed = not (cfg.bid_only and side < 0)
            if allowed and not any(s.side == side for s in self.spoofs):
                size = max(cfg.lot, round(cfg.spoof_k * self.stats.touch_depth[t] / cfg.lot) * cfg.lot)
                price = d.bid_price[t, 0] if side > 0 else d.ask_price[t, 0]
                self.spoofs.append(Spoof(side, float(size), float(price)))
        elif action == CANCEL:
            self.spoofs.clear()

        # advance the historical tape and check whether resting spoofs were run over
        fills = []
        for _ in range(cfg.events_per_step):
            self.t += 1
            while self.t < len(d) - 1 and not self._valid[self.t]:
                self.t += 1
            fills += self._run_over(self.t)

        pnl = self.pnl()
        scale = self.stats.ref_spread[self.t] * cfg.lot
        reward = (pnl - self._prev_pnl) / scale - cfg.inv_penalty * abs(self.inventory) / cfg.lot
        self._prev_pnl = pnl
        self.steps += 1

        terminated = abs(self.inventory) > self.max_inventory
        truncated = self.steps >= cfg.episode_len or self.t >= len(d) - 2
        return self._obs(), float(reward), terminated, truncated, self._info(fills)

    def _run_over(self, t: int) -> list[dict]:
        d, fills, keep = self.day, [], []
        for s in self.spoofs:
            crossed = (s.side > 0 and d.ask_price[t, 0] <= s.price) or (s.side < 0 and d.bid_price[t, 0] >= s.price)
            if crossed:
                self.inventory += int(s.side * s.size)
                self.cash -= s.side * s.size * s.price
                fills.append({"side": s.side, "size": s.size, "price": s.price, "t": t})
            else:
                keep.append(s)
        self.spoofs = keep
        return fills


def make_env(ticker: str, **overrides) -> LimitOrderBookEnv:
    return LimitOrderBookEnv(replace(EnvConfig(ticker=ticker), **overrides))
