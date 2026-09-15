import numpy as np
import pytest

from env.lobster_data import LobsterDay
from env.normalization import ReferenceStats
from evaluation.baseline_detector import (OrderRecord, RuleDetector, classification_metrics,
                                          real_data_flags, tune)


def _order(mult, lifetime, removed_by="cancel", trades=0):
    return OrderRecord(0, 1, mult * 100, mult, 10, 10 + lifetime, removed_by, trades)


def test_rule_requires_size_cancel_and_speed():
    det = RuleDetector(5, 50)
    assert det.flags(_order(10, 20))
    assert not det.flags(_order(3, 20))              # too small
    assert not det.flags(_order(10, 80))             # rested too long
    assert not det.flags(_order(10, 20, "run_over"))  # executed, not cancelled
    assert not det.flags(_order(10, 20, "episode_end"))


def test_rule_measures_lifetime_in_market_events():
    """An order resting 10 agent steps at 10 events/step lived 100 events, beyond M=50."""
    det = RuleDetector(5, 50)
    fast = OrderRecord(0, 1, 1000, 10, 0, 10, "cancel", 0, events_per_step=1)
    coarse = OrderRecord(0, 1, 1000, 10, 0, 10, "cancel", 0, events_per_step=10)
    assert fast.lifetime == coarse.lifetime == 10
    assert coarse.lifetime_events == 100
    assert det.flags(fast) and not det.flags(coarse)


def test_metrics_and_tuning():
    orders = [_order(10, 20, trades=2), _order(10, 20, trades=0), _order(10, 400, trades=1), _order(1, 5)]
    m = classification_metrics(orders, RuleDetector(5, 50))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    det, _ = tune(orders)
    assert classification_metrics(orders, det)["f1"] >= m["f1"]


def test_real_data_rule_on_synthetic_messages():
    # order 1: large, deleted after 3 events -> flag; order 2: large, executed before delete -> no flag;
    # order 3: small -> no flag; order 4: large, deleted after 100 events -> no flag (M=50)
    n = 200
    et = np.full(n, 1, dtype=np.int8)
    oid = np.arange(1000, 1000 + n)
    size = np.full(n, 10)
    rows = {0: (1, 1, 1000), 3: (3, 1, 100), 5: (1, 2, 1000), 6: (4, 2, 10), 8: (3, 2, 990),
            10: (1, 3, 10), 12: (3, 3, 10), 20: (1, 4, 1000), 120: (3, 4, 1000)}
    for i, (e, o, s) in rows.items():
        et[i], oid[i], size[i] = e, o, s
    z = np.zeros((n, 1))
    day = LobsterDay("SYN", 1, np.arange(n, dtype=float), et, oid, size, np.ones(n), np.ones(n, dtype=np.int8),
                     z + 2, z + 100, z + 1, z + 100)
    stats = ReferenceStats(np.ones(n), z + 100, z + 100)
    out = real_data_flags(day, stats, RuleDetector(5, 50), warmup=0)
    assert out["large_orders"] == 3
    assert out["flags"] == 1
