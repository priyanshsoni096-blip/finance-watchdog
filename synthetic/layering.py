"""LAYER-ATK: an evaluation-only layering attacker for the synthetic market.

The replay environment allows one large order per side at the best price, so the layering seen in real cases
(Coscia's progressively priced orders, Sarao's orders held several levels back) cannot be simulated there. The
synthetic market has a real order book, so this attacker can rest several large orders at once:

  * place `layers` large orders on one side at the best price and the next `layers - 1` price levels behind it,
    each (total_depth_mult / layers) x the trailing touch depth, so the total matches the usual 10x spoof
  * wait 30-80 market events
  * trade 1-5 lots on the opposite side
  * cancel every layer, then unwind the position

It needs `SyntheticSpoofEnv.place_large_order`, so it only runs in the synthetic market. Nothing is trained on it.
"""
from __future__ import annotations

from env.lob_env import BUY, CANCEL, NOOP, SELL


class LayerSpoof:
    def __init__(self, layers: int = 4, total_depth_mult: float = 10.0):
        self.layers = layers
        self.total_depth_mult = total_depth_mult

    def reset(self, rng):
        self.plan = []

    def act(self, obs, env, rng):
        if self.plan:
            return self.plan.pop(0)
        if env.inventory != 0:
            return SELL if env.inventory > 0 else BUY
        per_step = env.cfg.events_per_step
        if env.spoofs or rng.random() >= 0.01 * per_step:
            return NOOP
        side = 1 if rng.random() < 0.5 else -1
        book = env.market.book
        touch = book.best_bid() if side > 0 else book.best_ask()
        if touch is None:
            return NOOP
        per_layer = self.total_depth_mult / self.layers * env.stats.touch_depth[env.t]
        size = max(env.lot, int(round(per_layer / env.lot)) * env.lot)
        placed = sum(env.place_large_order(side, touch - side * i, size) for i in range(self.layers))
        if placed == 0:
            return NOOP
        trade = SELL if side > 0 else BUY
        wait = -(-int(rng.integers(30, 81)) // per_step)
        self.plan = [NOOP] * max(wait - 1, 0) + [trade] * int(rng.integers(1, 6)) + [CANCEL]
        return NOOP
