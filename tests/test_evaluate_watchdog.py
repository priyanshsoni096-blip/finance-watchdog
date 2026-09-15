import math

import numpy as np

from evaluation.baseline_detector import OrderRecord
from watchdog.evaluate_watchdog import counts, order_flags


def _o(placed, removed, trades=0):
    return OrderRecord(0, 1, 1000, 10, placed, removed, "cancel", trades)


def test_order_flags_use_resting_window_and_delay():
    flags = [np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)]
    orders = [(0, _o(0, 5, 1)), (0, _o(4, 7)), (0, _o(3, 4))]
    dec, delay = order_flags(orders, flags)
    assert dec == [True, False, True]
    assert delay[0] == 2 and math.isnan(delay[1]) and delay[2] == 0
    # step 7 is outside [4, 7): the second order is not credited with that flag


def test_counts():
    c = counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5 and c["fpr"] == 0.5
    empty = counts([0, 0], [0, 0])
    assert math.isnan(empty["recall"]) and empty["fpr"] == 0.0
