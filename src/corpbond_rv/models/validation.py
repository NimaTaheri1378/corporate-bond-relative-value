from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _num_samples(x) -> int:
    if hasattr(x, "shape") and x.shape:
        return int(x.shape[0])
    return len(x)


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Walk-forward splitter with an embargo gap between train and test."""

    n_splits: int = 5
    test_size: int | None = None
    gap: int = 0
    min_train_size: int | None = None
    expanding: bool = True

    def __post_init__(self) -> None:
        if self.n_splits <= 0:
            raise ValueError("n_splits must be positive")
        if self.test_size is not None and self.test_size <= 0:
            raise ValueError("test_size must be positive")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.min_train_size is not None and self.min_train_size <= 0:
            raise ValueError("min_train_size must be positive")

    def split(self, x, y=None, groups=None):
        n = _num_samples(x)
        test_size = self.test_size or max(1, n // (self.n_splits + 2))
        min_train = self.min_train_size or max(test_size, self.gap + 1)
        first_test_start = min_train + self.gap

        for split_idx in range(self.n_splits):
            test_start = first_test_start + split_idx * test_size
            test_end = min(test_start + test_size, n)
            train_end = test_start - self.gap

            if test_start >= n or test_end <= test_start:
                break
            if train_end < min_train:
                continue

            train_start = 0 if self.expanding else max(0, train_end - min_train)
            train_idx = np.arange(train_start, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)

            if train_idx.size == 0 or test_idx.size == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
        return self.n_splits
