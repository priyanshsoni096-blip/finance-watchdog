"""The per-instrument split is a reproducible observation with no established cause.

`results/detectability_by_instrument.md` states that two mechanical explanations were tested and refuted. A
claim like that rots quietly: if a future change to the features or the attacker made one of those mechanisms
explanatory after all, the document would still say "refuted". These tests fail in that case, so the document
has to be rewritten rather than silently becoming wrong.
"""
import numpy as np
import pytest

from scripts.detectability_by_instrument import DEEP_IMBALANCE, corr, measure, per_instrument_f1
from watchdog.dataset import DATA_DIR

V2 = DATA_DIR.parent / "watchdog_v2"
RESULTS = DATA_DIR.parent.parent / "results"

pytestmark = pytest.mark.skipif(
    not (V2 / "heldout").exists() or not (RESULTS / "anonymised_experiment.json").exists(),
    reason="temporal-split data or the attribution-free experiment has not been generated")


@pytest.fixture(scope="module")
def rows():
    import json
    cal = json.loads((RESULTS.parent / "configs" / "calibration.json").read_text(encoding="utf-8"))["tickers"]
    f1 = per_instrument_f1(RESULTS)
    if not f1:
        pytest.skip("no per-source results recorded")
    out = measure(V2, cal, f1)
    if len(out) < 3:
        pytest.skip("too few scripted-attacker sources to compare")
    return out


def test_the_split_is_real(rows):
    """If detectability stops varying by instrument, the whole document is moot."""
    f1 = [r["step_f1"] for r in rows if np.isfinite(r["step_f1"])]
    assert max(f1) - min(f1) > 0.3, f"per-instrument F1 range collapsed to {max(f1) - min(f1):.3f}"


def test_h1_attack_size_is_identical_across_instruments(rows):
    """The refutation of H1 rests on the attacker being the same size everywhere."""
    mult = [r["spoof_size_over_depth"] for r in rows if np.isfinite(r["spoof_size_over_depth"])]
    assert max(mult) - min(mult) < 0.1, (
        f"spoof size now varies by instrument ({min(mult):.2f}-{max(mult):.2f}); H1 is no longer refuted and "
        "results/detectability_by_instrument.md must be rewritten")


def test_h2_book_shift_does_not_explain_detectability(rows):
    c = corr([r["book_imbalance_shift"] for r in rows], [r["step_f1"] for r in rows])
    assert c < 0.5, (
        f"book-imbalance shift now correlates {c:.3f} with detectability; H2 is no longer refuted and the "
        "write-up must be rewritten")


def test_h2_signal_to_noise_does_not_explain_detectability(rows):
    c = corr([r["signal_to_noise"] for r in rows], [r["step_f1"] for r in rows])
    assert c < 0.5, (
        f"signal-to-noise now correlates {c:.3f} with detectability; H2 is no longer refuted")


def test_the_spread_association_is_reported_as_descriptive_only(rows):
    """Five instruments cannot support a mechanism, however strong the correlation looks."""
    assert len(rows) <= 10, "with more instruments this should become a real test, not a description"
    c = corr([r["median_spread_ticks"] for r in rows], [r["step_f1"] for r in rows])
    assert np.isfinite(c), "the spread association should still be computable"


def test_corr_refuses_to_report_on_too_few_points():
    assert not np.isfinite(corr([1.0, 2.0], [1.0, 2.0])), "two points must not yield a correlation"
    assert not np.isfinite(corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])), "zero variance must not yield one either"


def test_deep_imbalance_column_is_the_normalised_one(rows):
    """DEEP_IMBALANCE indexes book_features(); a reorder there would silently change every number here."""
    from watchdog.anonymised_experiment import DERIVED_NAMES
    assert DERIVED_NAMES[DEEP_IMBALANCE] == "deep imbalance (normalised)"
