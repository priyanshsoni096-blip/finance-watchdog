import math

import pytest

from watchdog.multiseed_summary import T_975, mean_ci95


def test_ci_uses_student_t_for_three_seeds():
    mean, half = mean_ci95([0.70, 0.75, 0.80])
    assert mean == pytest.approx(0.75)
    # sd (ddof=1) = 0.05, t(0.975, 2) = 4.303, half = 4.303 * 0.05 / sqrt(3)
    assert half == pytest.approx(4.303 * 0.05 / math.sqrt(3))


def test_ci_ignores_nan_and_handles_small_samples():
    mean, half = mean_ci95([0.5, float("nan")])
    assert mean == pytest.approx(0.5) and math.isnan(half)
    mean, half = mean_ci95([])
    assert math.isnan(mean) and math.isnan(half)


def test_t_table_decreases_with_sample_size():
    values = [T_975[n] for n in sorted(T_975)]
    assert values == sorted(values, reverse=True)
