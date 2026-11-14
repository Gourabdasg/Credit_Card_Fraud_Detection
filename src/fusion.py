"""Fusion strategies: simple averaging (Review-2) and stacked meta-classifier (Review-3)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def simple_average(probs: list) -> np.ndarray:
    """Plain average of branch fraud probabilities (decision threshold 0.5 applied later)."""
    return np.mean(np.vstack(probs), axis=0)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the decision threshold that maximises F1 on the validation split."""
    candidates = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


class StackedFusion:
    """Class-weighted logistic regression over branch probabilities + tuned threshold.

    Fit on the validation split: the meta-classifier learns how much to trust each branch, then
    the threshold is chosen on the same split to maximise F1 (instead of a fixed 0.5).
    """

    def __init__(self, branch_names):
        self.branch_names = list(branch_names)
        self.meta = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.threshold = 0.5

    def fit(self, val_probs: list, y_val: np.ndarray) -> "StackedFusion":
        Z = np.column_stack(val_probs)
        self.meta.fit(Z, y_val)
        self.threshold = tune_threshold(y_val, self.meta.predict_proba(Z)[:, 1])
        return self

    def predict_proba(self, probs: list) -> np.ndarray:
        return self.meta.predict_proba(np.column_stack(probs))[:, 1]

    def predict(self, probs: list) -> np.ndarray:
        return (self.predict_proba(probs) >= self.threshold).astype(int)

    @property
    def weights(self) -> dict:
        return dict(zip(self.branch_names, self.meta.coef_[0].round(3)))
