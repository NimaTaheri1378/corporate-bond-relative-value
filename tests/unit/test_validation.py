import numpy as np

from corpbond_rv.models.validation import PurgedWalkForwardSplit


def test_purged_walk_forward_split_has_embargo_gap():
    x = np.arange(80)
    splitter = PurgedWalkForwardSplit(n_splits=3, test_size=10, gap=5, min_train_size=20)
    splits = list(splitter.split(x))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) >= 20
        assert len(test_idx) > 0
        assert train_idx.max() + 5 < test_idx.min()
