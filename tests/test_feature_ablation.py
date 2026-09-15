import numpy as np

from evaluation.rollout import BOOK_DIM, OBS_DIM
from watchdog.feature_ablation import GROUPS, ablate


def test_groups_are_valid_and_well_formed():
    for name, idx in GROUPS.items():
        assert idx, name
        assert len(set(idx)) == len(idx), name
        assert all(0 <= i < OBS_DIM for i in idx), name
    assert sorted(GROUPS["book price offsets (all levels)"] + GROUPS["book sizes (all levels)"]) == list(range(BOOK_DIM))
    assert GROUPS["all participant features (6)"] == list(range(BOOK_DIM, OBS_DIM))
    singles = [v[0] for k, v in GROUPS.items() if k.startswith("participant:")]
    assert sorted(singles) == list(range(BOOK_DIM, OBS_DIM))


def test_ablate_replaces_only_chosen_columns():
    X = np.arange(3 * OBS_DIM, dtype=np.float32).reshape(3, OBS_DIM)
    fill = np.full(OBS_DIM, -7.0, dtype=np.float32)
    out = ablate(X, [0, 44], fill)
    assert np.all(out[:, [0, 44]] == -7.0)
    untouched = [i for i in range(OBS_DIM) if i not in (0, 44)]
    assert np.array_equal(out[:, untouched], X[:, untouched])
    assert X[0, 0] == 0.0            # the input is not modified in place
