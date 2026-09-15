"""SyntheticSpoofEnv: the frozen agents' interface on top of the synthetic agent-based market (evaluation only).

It exposes the same observation (43 dims), actions and the attributes that trained Spoofers, scripted agents,
StepTracker and the Watchdog rely on (`t`, `cfg`, `stats.touch_depth[t]`, `stats.ref_spread[t]`, `spoofs`,
`inventory`, `lot`), but prices come from the synthetic market instead of historical replay:

  * The agent's large orders really rest in the book, so imbalance followers see them and they can be filled
    by the market. No impact formula is applied.
  * The agent's market orders walk the book (self-impact emerges); exchange-style self-trade prevention stops
    them trading against the agent's own resting orders.
  * Reference statistics (trailing median spread, EWMA level sizes) follow env/normalization.py and exclude
    the agent's own resting volume, as in the replay environment, where historical sizes never contain it.
  * Inventory is marked to the market mid; remaining inventory is liquidated with a market order at the end.

Nothing is trained in this environment.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.lob_env import BUY, CANCEL, SELL, SPOOF_BUY, SPOOF_SELL, EnvConfig
from env.normalization import AGENT_CLIP, PRICE_CLIP, SIZE_CLIP, SIZE_HALFLIFE, SPREAD_WINDOW, encode_agent
from synthetic.market import BUY as MBUY
from synthetic.market import SELL as MSELL
from synthetic.simulator import MarketParams, SyntheticMarket

AGENT = "agent"


@dataclass
class LiveSpoof:
    order_id: int
    side: int          # +1 bid, -1 ask
    size: float        # remaining shares
    price: float       # dollars
    placed_t: int
    initial_size: float = 0.0


class OnlineReferenceStats:
    """Causal reference statistics equivalent to env.normalization.reference_stats, updated event by event."""

    def __init__(self, levels: int, spread_window: int = SPREAD_WINDOW, size_halflife: int = SIZE_HALFLIFE):
        self.alpha = 1.0 - 0.5 ** (1.0 / size_halflife)
        self.ask_ewma = np.full(levels, np.nan)
        self.bid_ewma = np.full(levels, np.nan)
        self._window: deque = deque(maxlen=spread_window)
        self._counts: Counter = Counter()
        self.touch_depth: list[float] = []
        self.ref_spread: list[float] = []   # ticks

    def _ewma(self, state: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        valid = sizes > 0
        first = valid & np.isnan(state)
        state[first] = sizes[first]
        rest = valid & ~first
        state[rest] = (1.0 - self.alpha) * state[rest] + self.alpha * sizes[rest]
        return state

    def refs(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.clip(np.nan_to_num(self.ask_ewma, nan=1.0), 1.0, None),
                np.clip(np.nan_to_num(self.bid_ewma, nan=1.0), 1.0, None))

    def update(self, ask_size: np.ndarray, bid_size: np.ndarray, spread_ticks: float) -> None:
        self._ewma(self.ask_ewma, ask_size)
        self._ewma(self.bid_ewma, bid_size)
        if len(self._window) == self._window.maxlen:
            old = self._window[0]
            self._counts[old] -= 1
        self._window.append(spread_ticks)
        self._counts[spread_ticks] += 1
        self.ref_spread.append(self._median())
        ask_ref, bid_ref = self.refs()
        self.touch_depth.append(float((ask_ref[0] + bid_ref[0]) / 2.0))

    def _median(self) -> float:
        n = len(self._window)
        values = sorted(v for v, c in self._counts.items() if c > 0)
        lo_target, hi_target = (n - 1) // 2, n // 2
        seen, lo, hi = 0, None, None
        for v in values:
            c = self._counts[v]
            if lo is None and seen + c > lo_target:
                lo = v
            if seen + c > hi_target:
                hi = v
                break
            seen += c
        return float((lo + hi) / 2.0)


def encode_live_book(ask_p, ask_s, bid_p, bid_s, ask_ref, bid_ref, ref_spread) -> np.ndarray:
    """Same encoding as env.normalization.encode_book for one row (prices and spread in the same units)."""
    mid = (ask_p[0] + bid_p[0]) / 2.0
    ask_pf = np.nan_to_num((ask_p - mid) / ref_spread, nan=PRICE_CLIP)
    bid_pf = np.nan_to_num((bid_p - mid) / ref_spread, nan=-PRICE_CLIP)
    ask_sf = np.log1p(ask_s / ask_ref)
    bid_sf = np.log1p(bid_s / bid_ref)
    out = np.stack([np.clip(ask_pf, -PRICE_CLIP, PRICE_CLIP), np.clip(ask_sf, 0.0, SIZE_CLIP),
                    np.clip(bid_pf, -PRICE_CLIP, PRICE_CLIP), np.clip(bid_sf, 0.0, SIZE_CLIP)], axis=-1)
    return out.reshape(-1).astype(np.float32)


class SyntheticSpoofEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None, params: MarketParams | None = None,
                 warmup_events: int = 6_000, blind_followers: bool = False):
        super().__init__()
        self.cfg = config if isinstance(config, EnvConfig) else EnvConfig(**(config or {}))
        self.params = params or MarketParams()
        self.warmup_events = warmup_events
        self.blind_followers = blind_followers   # counterfactual: followers ignore the agent's resting orders
        levels = self.cfg.levels
        low = np.concatenate([np.tile([-PRICE_CLIP, 0.0], 2 * levels), np.full(3, -AGENT_CLIP)])
        high = np.concatenate([np.tile([PRICE_CLIP, SIZE_CLIP], 2 * levels), np.full(3, AGENT_CLIP)])
        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.market: SyntheticMarket | None = None

    # ------------------------------------------------------------ helpers
    @property
    def tick(self) -> float:
        return self.params.tick_value

    def _agent_volume(self, side: int, price_ticks: float) -> float:
        return sum(s.size for s in self.spoofs if s.side == side and abs(s.price / self.tick - price_ticks) < 1e-9)

    def _record_event(self) -> None:
        book = self.market.book
        ask_p, ask_s, bid_p, bid_s = book.snapshot(self.cfg.levels)
        # reference statistics exclude the agent's own resting volume
        ask_s_hist = ask_s - np.array([self._agent_volume(-1, p) if np.isfinite(p) else 0.0 for p in ask_p])
        bid_s_hist = bid_s - np.array([self._agent_volume(1, p) if np.isfinite(p) else 0.0 for p in bid_p])
        if np.isfinite(ask_p[0]) and np.isfinite(bid_p[0]):
            spread = ask_p[0] - bid_p[0]
        else:                                   # one-sided book: carry the last reference spread (1 tick at start)
            spread = self.stats.ref_spread[-1] if self.stats.ref_spread else 1.0
        self.stats.update(np.clip(ask_s_hist, 0, None), np.clip(bid_s_hist, 0, None), float(spread))

    def mid_dollars(self) -> float:
        mid = self.market.book.mid()
        return float((mid if mid is not None else self.market.last_mid) * self.tick)

    def pnl(self) -> float:
        return self.cash + self.inventory * self.mid_dollars()

    @property
    def spoofing_active(self) -> bool:
        return len(self.spoofs) > 0

    def _obs(self) -> np.ndarray:
        book = self.market.book
        ask_p, ask_s, bid_p, bid_s = book.snapshot(self.cfg.levels)
        ask_ref, bid_ref = self.stats.refs()
        ref_spread = max(self.stats.ref_spread[-1], 1e-9)
        if not (np.isfinite(ask_p[0]) and np.isfinite(bid_p[0])):
            mid = self.market.last_mid
            ask_p = ask_p.copy()
            bid_p = bid_p.copy()
            ask_p[0] = ask_p[0] if np.isfinite(ask_p[0]) else mid + 0.5
            bid_p[0] = bid_p[0] if np.isfinite(bid_p[0]) else mid - 0.5
        book_feats = encode_live_book(ask_p, ask_s, bid_p, bid_s, ask_ref, bid_ref, ref_spread)
        agent = encode_agent(self.inventory, self.pnl(), sum(s.size for s in self.spoofs), mid=self.mid_dollars(),
                             lot=self.lot, max_inventory=self.max_inventory, touch_depth=self.stats.touch_depth[self.t])
        return np.concatenate([book_feats, agent]).astype(np.float32)

    def _book_agent_fills(self, fills, t: int) -> list[dict]:
        """Apply fills in which the agent's resting order was the maker (it was run over by the market)."""
        run_over = []
        for f in fills:
            if f.maker_owner != AGENT:
                continue
            spoof = next((s for s in self.spoofs if s.order_id == f.maker_id), None)
            if spoof is None:
                continue
            price = f.price * self.tick
            self.inventory += int(spoof.side * f.size)
            self.cash -= spoof.side * f.size * price
            spoof.size -= f.size
            if spoof.size <= 0:
                self.spoofs.remove(spoof)
                run_over.append({"side": spoof.side, "size": spoof.initial_size, "price": spoof.price, "t": t})
        return run_over

    def place_large_order(self, side: int, price_ticks: int, size: int) -> bool:
        """Rest an additional large order for the agent (used by evaluation-only layering agents).

        Only non-marketable prices are accepted: a buy must be below the best ask and a sell above the best bid.
        The order is tracked like a spoof placed through the action space, so StepTracker, run-over booking and
        CANCEL all apply to it."""
        book = self.market.book
        if size <= 0:
            return False
        if side > 0 and book.best_ask() is not None and price_ticks >= book.best_ask():
            return False
        if side < 0 and book.best_bid() is not None and price_ticks <= book.best_bid():
            return False
        oid, _ = book.limit(MBUY if side > 0 else MSELL, int(price_ticks), int(size), AGENT)
        if oid is None:
            return False
        self.spoofs.append(LiveSpoof(oid, side, float(size), price_ticks * self.tick, self.t, float(size)))
        return True

    def _market_order(self, side: int, qty: int) -> tuple[int, float | None]:
        fills = self.market.book.market(MBUY if side > 0 else MSELL, qty, AGENT, self_trade_prevention=True)
        filled = sum(f.size for f in fills)
        if filled == 0:
            return 0, None
        notional = sum(f.size * f.price for f in fills) * self.tick
        self.inventory += side * filled
        self.cash -= side * notional
        return filled, notional / filled

    # ------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        market_seed = int(self.np_random.integers(2**31 - 1))
        self.market = SyntheticMarket(self.params, seed=market_seed)
        if self.blind_followers:
            self._blind_market()
        self.stats = OnlineReferenceStats(self.cfg.levels)
        self.spoofs: list[LiveSpoof] = []
        self.inventory, self.cash = 0, 0.0
        self.t = -1
        for _ in range(self.warmup_events):
            self.market.step()
            self._record_event()
            self.t += 1
        depth = float(np.median(self.stats.touch_depth[len(self.stats.touch_depth) // 2:]))
        self.lot = int(self.cfg.lot) if self.cfg.lot is not None else max(100, int(round(depth / 2 / 100)) * 100)
        self.max_inventory = self.cfg.max_inventory_lots * self.lot
        self.steps = 0
        self._prev_pnl = 0.0
        return self._obs(), self._info([], None, {})

    def _blind_market(self) -> None:
        """Counterfactual: imbalance followers compute imbalance without the agent's resting volume."""
        market, env = self.market, self

        def imbalance(levels=None):
            n = levels or market.p.follower_levels
            bid = sum(s - env._agent_volume(1, p) for p, s in market.book.levels(MBUY, n))
            ask = sum(s - env._agent_volume(-1, p) for p, s in market.book.levels(MSELL, n))
            return 0.0 if bid + ask <= 0 else (bid - ask) / (bid + ask)

        market.imbalance = imbalance

    def _info(self, run_over, trade_price, liquidation) -> dict:
        return {"t": self.t, "pnl": self.pnl(), "inventory": self.inventory, "cash": self.cash,
                "spoofing_active": self.spoofing_active, "impact_shift": 0.0, "self_shift": 0.0,
                "trade_price": trade_price, "run_over": list(run_over), "liquidation": liquidation}

    def step(self, action):
        action = int(action)
        cfg, lot = self.cfg, self.lot
        trade_price = None
        book = self.market.book
        if action == BUY and self.inventory + lot <= self.max_inventory:
            _, trade_price = self._market_order(1, lot)
        elif action == SELL and self.inventory - lot >= -self.max_inventory:
            _, trade_price = self._market_order(-1, lot)
        elif action in (SPOOF_BUY, SPOOF_SELL):
            side = 1 if action == SPOOF_BUY else -1
            allowed = not (cfg.bid_only and side < 0)
            touch = book.best_bid() if side > 0 else book.best_ask()
            if allowed and touch is not None and not any(s.side == side for s in self.spoofs):
                size = max(lot, round(cfg.spoof_k * self.stats.touch_depth[self.t] / lot) * lot)
                oid, fills = book.limit(MBUY if side > 0 else MSELL, touch, int(size), AGENT)
                if oid is not None:
                    self.spoofs.append(LiveSpoof(oid, side, float(size), touch * self.tick, self.t, float(size)))
        elif action == CANCEL:
            for s in self.spoofs:
                book.cancel(s.order_id)
            self.spoofs.clear()

        run_over = []
        for _ in range(cfg.events_per_step):
            fills = self.market.step()
            self.t += 1
            run_over += self._book_agent_fills(fills, self.t)
            self._record_event()

        self.steps += 1
        truncated = self.steps >= cfg.episode_len
        liquidation = {}
        if truncated:
            for s in self.spoofs:
                book.cancel(s.order_id)
            self.spoofs.clear()
            if self.inventory != 0:
                qty = self.inventory
                filled, price = self._market_order(-1 if qty > 0 else 1, abs(qty))
                liquidation = {"qty": qty, "filled": filled, "price": price}
        pnl = self.pnl()
        scale = max(self.stats.ref_spread[-1], 1e-9) * self.tick * lot
        reward = (pnl - self._prev_pnl) / scale - cfg.inv_penalty * abs(self.inventory) / lot
        self._prev_pnl = pnl
        return self._obs(), float(reward), False, truncated, self._info(run_over, trade_price, liquidation)
