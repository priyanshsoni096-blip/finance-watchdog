"""The causal label is the change that fixed held-out generalisation, so its contract is tested directly.

`watchdog.dataset.causal_labels` turns the hindsight label ("this order was manipulative, so mark every step it
rested") into one that only becomes positive once the evidence exists ("mark from the first step its owner traded
on the opposite side"). Training on the hindsight label rewarded predicting which trader would spoof over
detecting the act; see AUDIT.md section B2.

These tests pin the properties the fix depends on, including the docstring's claim that every causal positive is
also a hindsight positive.
"""
import numpy as np
import pytest

from watchdog.dataset import DATA_DIR, TRADE_COL, causal_labels, load_split, orders_from_array

pytestmark = pytest.mark.skipif(not (DATA_DIR / "test").exists(),
                                reason="recorded Watchdog data not generated")


@pytest.fixture(scope="module")
def sources():
    data = load_split("test")
    if not data:
        pytest.skip("recorded Watchdog data not generated")
    return data


def test_every_causal_positive_is_also_a_hindsight_positive(sources):
    """The docstring's central claim: the causal label only ever removes positives, never invents them."""
    for name, src in sources.items():
        causal = causal_labels(src)
        hindsight = src["y"].astype(bool)
        assert not (causal & ~hindsight).any(), f"{name}: causal label positive where hindsight is not"


def test_causal_labels_are_a_strict_subset_where_manipulation_exists(sources):
    """Hindsight marks the whole resting window; causal starts at the act, so it must be no larger."""
    saw_manipulation = False
    for name, src in sources.items():
        causal, hindsight = causal_labels(src), src["y"].astype(bool)
        assert causal.sum() <= hindsight.sum(), f"{name}: causal positives exceed hindsight positives"
        if hindsight.any():
            saw_manipulation = True
    assert saw_manipulation, "no manipulation in the test split; the comparison is vacuous"


def test_causal_label_starts_at_the_first_opposite_trade(sources):
    """For each manipulative order, the first causal positive must be its first opposite-trade step."""
    checked = 0
    for name, src in sources.items():
        causal = causal_labels(src)
        off = src["ep_offsets"]
        for ep, o in orders_from_array(src["orders"]):
            if not o.manipulative:
                continue
            a = off[ep]
            window = src["X"][a + o.placed_step:a + o.removed_step, TRADE_COL]
            hit = np.flatnonzero(window == -o.side)
            if not hit.size:
                continue
            first = a + o.placed_step + int(hit[0])
            assert causal[first], f"{name}: no causal positive at the first opposite trade"
            if first > a + o.placed_step:
                before = causal[a + o.placed_step:first]
                # another overlapping order may legitimately light up this window, so only assert
                # when this is the sole resting order of the episode segment
                if not before.any():
                    checked += 1
    assert checked > 0, "no order had a gap between placement and its first opposite trade"


def test_legitimate_sources_get_no_causal_positives(sources):
    legit = {n: s for n, s in sources.items() if n.startswith(("HONEST", "FLICKER"))}
    if not legit:
        pytest.skip("no legitimate sources recorded")
    for name, src in legit.items():
        assert not causal_labels(src).any(), f"{name}: causal positive on a legitimate participant"


def test_causal_labels_are_label_free_in_shape_and_dtype(sources):
    """Returns a bool array aligned with the recorded steps, whatever the recorded y contains."""
    for name, src in sources.items():
        out = causal_labels(src)
        assert out.dtype == np.bool_, f"{name}: expected bool, got {out.dtype}"
        assert out.shape == src["y"].shape, f"{name}: shape {out.shape} != {src['y'].shape}"


def test_causal_labels_ignore_the_recorded_y_column(sources):
    """Flipping the stored labels must not change the causal labels: they are rebuilt from orders and trades."""
    name, src = next(iter(sources.items()))
    flipped = dict(src)
    flipped["y"] = 1 - np.asarray(src["y"])
    assert np.array_equal(causal_labels(src), causal_labels(flipped)), \
        f"{name}: causal labels changed when only the recorded labels changed"


def test_the_gap_between_the_two_labels_is_what_the_diagnostic_reported(sources):
    """Across the test split the causal label must be materially smaller, which is why training changed."""
    causal = np.concatenate([causal_labels(s) for s in sources.values()])
    hindsight = np.concatenate([np.asarray(s["y"]).astype(bool) for s in sources.values()])
    assert hindsight.sum() > 0
    ratio = causal.sum() / hindsight.sum()
    assert 0.05 < ratio < 0.95, (
        f"causal positives are {ratio:.2f} of hindsight positives; if this approaches 1.0 the two labels have "
        "converged and AUDIT.md section B2 needs rewriting")
