import numpy as np

from env.lob_env import EnvConfig
from evaluation.rollout import record_episode
from synthetic.layering import LayerSpoof
from synthetic.synthetic_env import AGENT, SyntheticSpoofEnv


def _env(episode_len=60):
    return SyntheticSpoofEnv(EnvConfig(events_per_step=10, episode_len=episode_len), warmup_events=2_000)


class _TriggerRng:
    """Stub RNG for the attacker: always triggers a placement on the bid side, with the shortest wait and one trade."""

    def random(self):
        return 0.0

    def integers(self, low, high):
        return low


def test_place_large_order_rests_non_marketable_orders_only():
    env = _env()
    env.reset(seed=1)
    book = env.market.book
    bb, ba = book.best_bid(), book.best_ask()
    assert env.place_large_order(1, bb - 2, 5_000)
    assert len(env.spoofs) == 1 and book.orders[env.spoofs[0].order_id].owner == AGENT
    assert abs(env.spoofs[0].price - (bb - 2) * env.tick) < 1e-12
    assert not env.place_large_order(1, ba, 5_000)       # a buy at the best ask would cross the spread
    assert not env.place_large_order(-1, bb, 5_000)      # a sell at the best bid would cross
    assert not env.place_large_order(1, bb - 1, 0)
    assert len(env.spoofs) == 1


def test_layer_attacker_places_layers_trades_and_cancels():
    env = _env()
    ep = record_episode(env, LayerSpoof(layers=4), _TriggerRng(), seed=5)
    first_step = min(o.placed_step for o in ep["orders"])
    first = sorted([o for o in ep["orders"] if o.placed_step == first_step], key=lambda o: o.order_id)
    assert len(first) == 4
    assert len({o.size for o in first}) == 1
    assert all(o.side == 1 for o in first)
    assert all(o.removed_by in ("cancel", "run_over") for o in first)
    assert any(o.manipulative for o in first)             # the opposite-side trade happened while they rested
