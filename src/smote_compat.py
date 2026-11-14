"""Compatibility wrapper for SMOTE.

The project's dependency on imbalanced-learn can fail on some Windows machines due to a
native DLL loading policy. In those environments, provide a small in-memory fallback that
keeps the project logic intact without importing the blocked extension modules.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from imblearn.over_sampling import SMOTE as _SMOTE
except Exception:  # pragma: no cover - environment-specific fallback
    _SMOTE = None


class SMOTE:
    """Minimal fallback that oversamples the minority class using duplication + tiny noise."""

    def __init__(self, random_state: int = 42, sampling_strategy: float = 1.0, *args, **kwargs):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X.copy(), y.copy()

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        majority_count = int(counts[np.argmax(counts)])
        minority_count = int(counts[np.argmin(counts)])

        target_majority = int(max(majority_count, minority_count * max(self.sampling_strategy, 1.0)))
        target_minority = int(minority_count)
        if majority_count <= minority_count:
            # Preserve the current class balance when the dataset is already balanced.
            return X.copy(), y.copy()

        needed = max(0, target_majority - minority_count)
        if needed == 0:
            return X.copy(), y.copy()

        rng = np.random.RandomState(self.random_state)
        minority_idx = np.flatnonzero(y == minority_class)
        chosen = rng.choice(minority_idx, size=needed, replace=True)
        synthetic = X[chosen].astype(np.float64, copy=True)
        synthetic += rng.normal(0.0, 1e-6, size=synthetic.shape)

        X_res = np.vstack([X.astype(np.float64, copy=False), synthetic])
        y_res = np.concatenate([y, np.full(needed, minority_class, dtype=y.dtype)])
        return X_res, y_res


if _SMOTE is not None:
    SMOTE = _SMOTE
