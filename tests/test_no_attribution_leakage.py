"""Guards for the attribution-free experiment: the participant block must never reach the model.

`watchdog/anonymised_experiment.py` exists to answer what is detectable WITHOUT privileged participant
attribution. That claim is only as good as the guarantee that features 40-45 (the participant block, which
includes the two that encode the label's definition) never enter the feature matrix by any route.
"""
import numpy as np
import pytest

from evaluation.rollout import BOOK_DIM, PARTICIPANT_DIM
from watchdog.anonymised_experiment import (DERIVED_NAMES, IMBALANCE_COL, TRADE_COL, book_features, build,
                                            feature_name, windowed)
from watchdog.dataset import DATA_DIR, load_split

pytestmark = pytest.mark.skipif(not (DATA_DIR / "train").exists(),
                                reason="recorded Watchdog data not generated")


@pytest.fixture(scope="module")
def train():
    d = load_split("train")
    if not d:
        pytest.skip("recorded Watchdog data not generated")
    return d


def test_withheld_columns_are_the_ones_that_encode_the_label():
    """Pins which columns are being withheld, so a change to the observation layout fails loudly."""
    assert TRADE_COL == BOOK_DIM + 4
    assert IMBALANCE_COL == BOOK_DIM + 5
    assert BOOK_DIM + PARTICIPANT_DIM == 46


def test_derived_features_depend_only_on_the_book(train):
    """Changing the participant block must not change a single derived feature."""
    src = next(iter(train.values()))
    X = src["X"].copy()
    before = book_features(X)
    tampered = X.copy()
    rng = np.random.default_rng(0)
    tampered[:, BOOK_DIM:] = rng.normal(size=(len(X), PARTICIPANT_DIM))
    after = book_features(tampered)
    assert np.array_equal(before, after), "a derived feature read the participant block"


def test_built_matrix_does_not_contain_the_participant_block(train):
    """The strongest check: no column of the model's input reproduces a withheld column."""
    got = build("train", DATA_DIR)
    assert got is not None
    X = got[0]
    src = next(iter(train.values()))
    withheld = src["X"][:, [TRADE_COL, IMBALANCE_COL]]
    n = min(len(X), len(withheld))
    for j in range(X.shape[1]):
        for k in range(withheld.shape[1]):
            assert not np.allclose(X[:n, j], withheld[:n, k], atol=1e-9), \
                f"retained column {j} ({feature_name(j)}) reproduces withheld column {k}"


def test_no_retained_feature_is_near_deterministic_for_the_label(train):
    """If a retained feature ever became label-equivalent, the experiment would be meaningless."""
    from watchdog.anonymised_experiment import leakage_audit
    a = leakage_audit(DATA_DIR)
    assert a["withheld_pair_step_f1"] > 0.8, "the withheld pair no longer encodes the label; rewrite the claim"
    assert a["max_abs_point_biserial_corr"] < 0.8, \
        f"a retained feature correlates {a['max_abs_point_biserial_corr']:.2f} with the label"
    assert a["best_single_feature_rule_f1"] < a["withheld_pair_step_f1"], \
        "a single retained feature matches the withheld pair; the removal is not real"


def test_feature_names_cover_every_column(train):
    got = build("train", DATA_DIR)
    X = got[0]
    base = BOOK_DIM + len(DERIVED_NAMES)
    assert X.shape[1] == base * 3, "windowing should triple [current, prev-mean, delta]"
    for j in range(base):
        name = feature_name(j)
        assert name and not name.startswith("derived["), f"column {j} has no readable name"
    assert feature_name(0) == "book L1 ask price"
    assert feature_name(BOOK_DIM) == DERIVED_NAMES[0]


def test_windowing_never_crosses_an_episode_boundary():
    """Rewriting one episode must leave every other episode's features untouched.

    This is the property that matters: the rolling window is computed per episode, so history cannot bleed
    across a boundary. (At an episode's first step the previous-window mean is 0 because there is no history
    yet, which is itself independent of whatever precedes it in the array.)
    """
    mat = np.arange(20, dtype=np.float32).reshape(10, 2)
    offsets = np.array([0, 4, 10])
    before = windowed(mat, offsets)

    tampered = mat.copy()
    tampered[0:4] = -999.0                      # rewrite episode 1 entirely
    after = windowed(tampered, offsets)

    assert np.array_equal(before[4:], after[4:]), "episode 2 changed when only episode 1 was rewritten"
    assert not np.array_equal(before[:4], after[:4]), "the tampering had no effect; the test proves nothing"
    # and the first step of each episode carries no history
    for a in (0, 4):
        assert np.allclose(before[a, 2:4], 0.0), f"step {a} has a previous-window mean without any history"


def test_episode_ids_are_unique_across_sources(train):
    """Episode-level aggregation must not merge episodes from different sources."""
    X, y, eps, owner = build("train", DATA_DIR)
    assert len(eps) == len(X) == len(y) == len(owner)
    for e in np.unique(eps)[:200]:
        assert len(set(owner[eps == e])) == 1, f"episode {e} spans more than one source"
