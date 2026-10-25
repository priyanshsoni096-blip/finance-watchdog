"""Statistics helpers added for the attribution-free experiment's fuller reporting.

Multi-seed intervals, PR-AUC, Brier score and the reliability curve all feed numbers straight into
`results/anonymised_experiment.md`, so each is pinned against a case whose answer can be worked out by hand.
"""
import numpy as np
import pytest

from watchdog.anonymised_experiment import (brier, episode_counts, pr_auc, reliability, student_t_ci)


# --------------------------------------------------------------------------- uncertainty
def test_single_seed_has_a_mean_but_no_interval():
    mu, ci = student_t_ci([0.42])
    assert mu == pytest.approx(0.42)
    assert ci != ci, "a single run cannot have a confidence interval"


def test_identical_seeds_give_a_zero_width_interval():
    mu, ci = student_t_ci([0.5, 0.5, 0.5])
    assert mu == pytest.approx(0.5) and ci == pytest.approx(0.0)


def test_interval_widens_with_spread():
    _, tight = student_t_ci([0.50, 0.51, 0.49])
    _, wide = student_t_ci([0.20, 0.50, 0.80])
    assert wide > tight > 0


def test_nans_are_dropped_not_propagated():
    mu, _ = student_t_ci([0.4, float("nan"), 0.6])
    assert mu == pytest.approx(0.5)
    assert student_t_ci([])[0] != student_t_ci([])[0], "empty input must be NaN, not an error"


def test_three_seed_interval_uses_the_small_sample_t_value():
    """n=3 must use t=4.303, not the normal 1.96, or intervals are reported too narrow."""
    vals = [0.0, 1.0, 2.0]
    mu, ci = student_t_ci(vals)
    expected = 4.303 * np.std(vals, ddof=1) / np.sqrt(3)
    assert mu == pytest.approx(1.0)
    assert ci == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- ranking and calibration
def test_pr_auc_is_one_for_a_perfect_ranking():
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert pr_auc(np.array([0.1, 0.2, 0.9, 0.8]), y) == pytest.approx(1.0)


def test_pr_auc_is_undefined_when_one_class_is_absent():
    assert pr_auc(np.array([0.1, 0.9]), np.array([0, 0], dtype=bool)) != \
        pr_auc(np.array([0.1, 0.9]), np.array([0, 0], dtype=bool))


def test_pr_auc_beats_the_base_rate_for_a_useful_score():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    score = np.where(y, rng.normal(0.7, 0.15, 2000), rng.normal(0.3, 0.15, 2000))
    assert pr_auc(score, y) > 0.5 > y.mean()


def test_brier_rewards_confident_correct_probabilities():
    y = np.array([1, 1, 0, 0])
    assert brier(np.array([1.0, 1.0, 0.0, 0.0]), y) == pytest.approx(0.0)
    assert brier(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.25)
    assert brier(np.array([0.0, 0.0, 1.0, 1.0]), y) == pytest.approx(1.0)


def test_reliability_bins_report_predicted_against_observed():
    score = np.array([0.05, 0.15, 0.95, 0.85])
    y = np.array([0, 0, 1, 1])
    bins = reliability(score, y, bins=10)
    assert bins, "no bins produced"
    assert all(b["n"] > 0 for b in bins), "an empty bin should be omitted, not reported"
    for b in bins:
        assert 0.0 <= b["predicted"] <= 1.0 and 0.0 <= b["observed"] <= 1.0
    assert sum(b["n"] for b in bins) == len(y), "every sample must land in exactly one bin"


def test_reliability_catches_a_miscalibrated_model():
    """A model that always says 0.9 on data that is 10% positive must show the gap."""
    score = np.full(100, 0.9)
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    b = reliability(score, y)[0]
    assert b["predicted"] == pytest.approx(0.9) and b["observed"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- episode aggregation
def test_episode_is_flagged_when_any_of_its_steps_is():
    pred = np.array([False, True, False, False, False, False])
    y = np.array([False, True, False, False, False, False])
    eps = np.array([0, 0, 0, 1, 1, 1])
    c = episode_counts(pred, y, eps)
    assert c["n"] == 2 and c["tp"] == 1 and c["tn"] == 1 and c["fp"] == 0 and c["fn"] == 0
