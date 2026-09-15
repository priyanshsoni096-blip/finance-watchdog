"""LimitOrderBookEnv — historical LOBSTER replay with an injected trading agent.

Actions (Discrete(6)):
  0 no-op | 1 market buy 1 lot | 2 market sell 1 lot
  3 spoof buy  (K x touch depth) at best bid | 4 spoof sell (K x touch depth) at best ask
  5 cancel all resting spoofs

Observation (Box(43), float32): 40 normalized book features (spread units / log size ratio,
resting spoofs added to the displayed size at their price level) + 3 agent features.

Economics:
  * Market orders fill at the IMPACTED touch: historical ask/bid + impact shift from resting spoofs
    + self-impact of the agent's own cumulative signed market-order volume (same calibrated lambda,
    no decay within an episode; each order pays the average impact of walking through its own size).
    Without self-impact a trained agent bought ~20x touch depth at a spoof-depressed price for free.
  * A spoof's impact builds up with its age following the measured order-flow response
    beta(W)/beta(200) (about half after 10 events on INTC, full at 200), so cancel-and-replace
    cannot refresh a full distortion. Each spoof's FAVOURABLE shift applies to at most its own size
    of opposite-side volume; adverse shifts always apply. Without these, trained agents swung
    +-40 lots under alternating spoofs for ~$240k/episode on MSFT/INTC.
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
from env.price_impact import PriceImpact, calibrate_lambda, impact_ramp

NOOP, BUY, SELL, SPOOF_BUY, SPOOF_SELL, CANCEL = range(6)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "configs" / "calibration.json"


@dataclass
class EnvConfig:
    ticker: str = "AAPL"
    levels: int = 10
    # None -> half the median touch depth, rounded to 100 shares (AAPL 100, MSFT ~6,700).
    # A fixed 100-share lot made a run-over on MSFT/INTC ~1,300 lots, impossible to unwind.
    lot: int | None = None
    spoof_k: float = 10.0              # spoof size as multiple of trailing touch depth (~20 lots)
    max_inventory_lots: int = 40       # limits market orders only; run-overs never end an episode
    inv_penalty: float = 0.001         # per lot held, per step, in reward units
    episode_len: int = 2_000           # agent steps
    events_per_step: int = 1
    # Events skipped before any episode may start. The opening book is thin: at event 15k INTC's
    # trailing touch depth is 0.08x its day median. With 30k, no start on any ticker yields a
    # spoof under 5 lots (measured; 3.9% of INTC starts did at 5k).
    warmup: int = 30_000
    impact_form: str = "linear"
    impact_lambda: float | None = None  # None -> per-ticker calibrated value
    bid_only: bool = False             # SPOOFER-03 constraint: spoof-sell disabled
    impact_ramp: bool = True           # spoof impact builds up with order age (measured)
    spoof_capacity: bool = True        # favourable spoof shift limited to the spoof's size in volume
    extra: dict = field(default_factory=dict)


@dataclass
class Spoof:
    side: int      # +1 buy (rests on bid), -1 sell (rests on ask)
    size: float
    price: float
    placed_t: int = 0
    capacity: float = 0.0  # opposite-side shares that can still trade at this spoof's favourable shift


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
        self._ramp = self._load_ramp()
        if cfg.lot is None:
            depth = float(np.median(self.stats.touch_depth[cfg.warmup:]))
            self.lot = max(100, int(round(depth / 2 / 100)) * 100)
        else:
            self.lot = int(cfg.lot)
        self.max_inventory = cfg.max_inventory_lots * self.lot

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

    def _load_ramp(self):
        if not self.cfg.impact_ramp:
            return None
        ramp = None
        if CALIBRATION_PATH.exists():
            cal = json.loads(CALIBRATION_PATH.read_text())
            entry = cal.get("tickers", {}).get(self.cfg.ticker, {})
            if cal.get("form") == self.cfg.impact_form and "ramp" in entry:
                ramp = {int(k): float(v) for k, v in entry["ramp"].items()}
        if ramp is None:
            ramp = impact_ramp(self.day, self.stats, self.cfg.impact_form)
        ws = sorted(ramp)
        return np.array([0.0] + ws, dtype=float), np.array([0.0] + [max(ramp[w], 0.0) for w in ws])

    # --------------------------------------------------------------- helpers
    def age_factor(self, spoof: Spoof, t: int) -> float:
        if self._ramp is None:
            return 1.0
        return float(np.interp(t - spoof.placed_t, *self._ramp))

    def _spoof_shift(self, spoof: Spoof, t: int) -> float:
        return self.impact.shift(spoof.side * spoof.size * self.age_factor(spoof, t),
                                 self.stats.touch_depth[t], self.stats.ref_spread[t])

    def impact_shift(self, t: int | None = None) -> float:
        """Total price shift from resting spoofs (before any capacity limit)."""
        t = self.t if t is None else t
        return sum(self._spoof_shift(s, t) for s in self.spoofs)

    def _fill_spoof_shift(self, direction: int, qty: float, t: int) -> float:
        """Spoof shift applied to an executed market order of `qty` shares (+1 buy / -1 sell).
        Adverse shifts apply in full. A favourable shift (buy under a spoof sell, sell under a spoof
        buy) applies only to the part of the order within that spoof's remaining capacity."""
        total = 0.0
        for s in self.spoofs:
            c = self._spoof_shift(s, t)
            if self.cfg.spoof_capacity and c * direction < 0:
                frac = min(1.0, s.capacity / qty)
                s.capacity = max(0.0, s.capacity - qty)
                c *= frac
            total += c
        return total

    def self_shift(self, volume: float | None = None, t: int | None = None) -> float:
        """Price shift from the agent's own signed market-order volume (buys positive)."""
        t = self.t if t is None else t
        v = self.volume if volume is None else volume
        return self.impact.shift(v, self.stats.touch_depth[t], self.stats.ref_spread[t])

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
                             mid=self.mark_mid(), lot=self.lot, max_inventory=self.max_inventory,
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
            "self_shift": self.self_shift(),
            "trade_price": self._last_trade,
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
        self.volume = 0.0          # signed shares traded by the agent's market orders this episode
        self._last_trade = None
        self._prev_pnl = 0.0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        cfg, d, t = self.cfg, self.day, self.t

        lot = self.lot
        self._last_trade = None
        if action == BUY and self.inventory + lot <= self.max_inventory:
            price = d.ask_price[t, 0] + self._fill_spoof_shift(1, lot, t) + self.self_shift(self.volume + lot / 2, t)
            self.inventory += lot
            self.volume += lot
            self.cash -= lot * price
            self._last_trade = float(price)
        elif action == SELL and self.inventory - lot >= -self.max_inventory:
            price = d.bid_price[t, 0] + self._fill_spoof_shift(-1, lot, t) + self.self_shift(self.volume - lot / 2, t)
            self.inventory -= lot
            self.volume -= lot
            self.cash += lot * price
            self._last_trade = float(price)
        elif action in (SPOOF_BUY, SPOOF_SELL):
            side = 1 if action == SPOOF_BUY else -1
            allowed = not (cfg.bid_only and side < 0)
            if allowed and not any(s.side == side for s in self.spoofs):
                size = max(lot, round(cfg.spoof_k * self.stats.touch_depth[t] / lot) * lot)
                price = d.bid_price[t, 0] if side > 0 else d.ask_price[t, 0]
                self.spoofs.append(Spoof(side, float(size), float(price), placed_t=t, capacity=float(size)))
        elif action == CANCEL:
            self.spoofs.clear()

        # advance the historical tape and check whether resting spoofs were run over
        fills = []
        for _ in range(cfg.events_per_step):
            self.t += 1
            while self.t < len(d) - 1 and not self._valid[self.t]:
                self.t += 1
            fills += self._run_over(self.t)

        self.steps += 1
        truncated = self.steps >= cfg.episode_len or self.t >= len(d) - 2
        liquidation = {}
        if truncated:
            liquidation = self._liquidate()

        pnl = self.pnl()
        scale = self.stats.ref_spread[self.t] * lot
        reward = (pnl - self._prev_pnl) / scale - cfg.inv_penalty * abs(self.inventory) / lot
        self._prev_pnl = pnl

        # No termination: ending the episode on a run-over let the agent escape ongoing costs
        # (measured: 31/31 early terminations happened on a run-over step).
        info = self._info(fills)
        info["liquidation"] = liquidation
        return self._obs(), float(reward), False, truncated, info

    def _liquidate(self) -> dict:
        """Close remaining position at episode end by crossing the unimpacted touch, so holding
        inventory to the end is not free (a mark-to-mid ending would ignore the spread)."""
        self.spoofs.clear()
        if self.inventory == 0:
            return {}
        t, d = self.t, self.day
        qty = self.inventory
        touch = d.bid_price[t, 0] if qty > 0 else d.ask_price[t, 0]
        # unwinding is a market order too: it pays the average self-impact of trading -qty shares
        price = touch + self.self_shift(self.volume - qty / 2, t)
        self.cash += qty * price
        self.volume -= qty
        self.inventory = 0
        return {"qty": qty, "price": float(price)}

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
