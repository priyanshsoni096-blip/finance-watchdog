"""The manipulation label is directly readable from two observation features (audit Phase 9).

This is the project's most important limitation, so it is locked down by a test rather than left in prose: if
a future change to the label or to the surveillance observation breaks the relationship, this test fails and
the claim in `results/label_observability.md` must be rewritten.

Nothing is fitted here. The signature is a comparison of feature 44 (the participant's trade direction) with
feature 45 (the side its resting orders sit on), which is the label's own definition.
"""
import numpy as np
import pytest

from scripts.label_observability import SIGNATURES, score
from watchdog.dataset import DATA_DIR, load_split

pytestmark = pytest.mark.skipif(not (DATA_DIR / "test").exists(),
                                reason="recorded Watchdog data not generated")

SIGNATURE = SIGNATURES["opposite-side signature"]


@pytest.fixture(scope="module")
def splits():
    test, held = load_split("test"), load_split("heldout")
    if not test or not held:
        pytest.skip("recorded Watchdog data not generated")
    return test, held


def test_signature_recovers_the_label_in_distribution(splits):
    test, _ = splits
    o = score(SIGNATURE, test)["orders"]
    assert o["positives"] > 0, "no manipulative orders to check"
    assert o["precision"] > 0.99, f"precision fell to {o['precision']:.3f}"
    assert o["recall"] > 0.95, f"recall fell to {o['recall']:.3f}"


def test_signature_transfers_to_the_held_out_scripted_attacker(splits):
    """A zero-parameter signature generalises, so a learned detector's failure to is about the learner."""
    _, held = splits
    scripted = {n: s for n, s in held.items() if n.startswith("SCRIPTED")}
    if not scripted:
        pytest.skip("no scripted attacker recorded")
    o = score(SIGNATURE, scripted)["orders"]
    assert o["precision"] > 0.99 and o["recall"] > 0.95, f"did not transfer: {o}"


def test_signature_never_fires_on_legitimate_participants(splits):
    test, held = splits
    legit = {n: s for n, s in (test | held).items() if n.startswith(("HONEST", "FLICKER"))}
    if not legit:
        pytest.skip("no legitimate sources recorded")
    b = score(SIGNATURE, legit)
    assert b["orders"]["positives"] == 0, "legitimate sources should hold no manipulative orders"
    assert b["orders"]["fp"] == 0, "the signature raised a false alarm on a legitimate participant"
    assert b["steps"]["fp"] == 0


def test_step_labels_extend_beyond_the_signature_by_design(splits):
    """Hindsight labels every resting step; the signature only fires where the trade happens."""
    test, _ = splits
    s = score(SIGNATURE, test)["steps"]
    assert s["precision"] > 0.99, "a step the signature fires on must be a labelled step"
    assert 0.5 < s["recall"] < 0.95, f"step recall {s['recall']:.3f} outside the expected hindsight gap"


def test_presence_of_a_resting_order_alone_is_not_enough(splits):
    """Guards against the trivial reading: it is the two-feature comparison that carries the label."""
    test, _ = splits
    o = score(SIGNATURES["resting order present"], test)["orders"]
    assert o["precision"] < 0.5, f"mere presence of an order should not identify manipulation (P={o['precision']:.3f})"


def test_signature_is_label_free(splits):
    """The signature may only read the observation, never the recorded label."""
    test, _ = splits
    src = next(iter(test.values()))
    X = src["X"].copy()
    flipped = dict(src)
    flipped["y"] = 1 - src["y"]
    before = SIGNATURE(X)
    after = SIGNATURE(flipped["X"])
    assert np.array_equal(before, after), "signature output changed when only labels changed"
