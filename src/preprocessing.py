"""Data preparation: stratified split, scaling, SMOTE (train only) and pseudo-sequence shaping."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from smote_compat import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    N_STEP_FEATURES,
    N_TIMESTEPS,
    SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

NUMERIC_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # 29 numeric features


@dataclass
class Prepared:
    # Numeric, shaped as (n, timesteps, step_features) for LSTM / CNN.
    X_train_seq: np.ndarray        # after SMOTE
    y_train: np.ndarray            # after SMOTE
    X_val_seq: np.ndarray
    y_val: np.ndarray
    X_test_seq: np.ndarray
    y_test: np.ndarray
    # Text for the NLP branch (original, un-resampled training split).
    text_train: pd.Series
    y_train_orig: np.ndarray
    text_val: pd.Series
    text_test: pd.Series
    # Bookkeeping for reporting.
    train_counts_before: tuple
    train_counts_after: tuple


def to_pseudo_sequence(X: np.ndarray) -> np.ndarray:
    """Zero-pad 29 features to 30 and reshape to (n, 5, 6).

    Kaggle rows are individual transactions, so this is only a pseudo-sequence. Ordering real
    transactions per cardholder / time is planned for later reviews.
    """
    n, f = X.shape
    total = N_TIMESTEPS * N_STEP_FEATURES
    if f > total:
        raise ValueError(f"Expected at most {total} features, got {f}")
    padded = np.zeros((n, total), dtype=np.float32)
    padded[:, :f] = X
    return padded.reshape(n, N_TIMESTEPS, N_STEP_FEATURES)


def prepare(df: pd.DataFrame, seed: int = SEED) -> Prepared:
    y = df["Class"].to_numpy()
    idx = np.arange(len(df))

    # Stratified 70 / 15 / 15 split.
    # Exact integer sizes (4200 / 900 / 900 for 6000 samples) avoid off-by-one rounding.
    n = len(df)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round(n * VAL_FRAC))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=n_val + n_test, stratify=y, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=n_test, stratify=y[idx_tmp], random_state=seed
    )

    X = df[NUMERIC_COLS].to_numpy(dtype=np.float64)

    # Scaling fitted on the training split only.
    scaler = StandardScaler().fit(X[idx_train])
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_test = scaler.transform(X[idx_test])

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    before = (int((y_train == 0).sum()), int((y_train == 1).sum()))

    # SMOTE on the training split only, so no synthetic sample leaks into val / test.
    X_train_res, y_train_res = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    after = (int((y_train_res == 0).sum()), int((y_train_res == 1).sum()))

    return Prepared(
        X_train_seq=to_pseudo_sequence(X_train_res),
        y_train=y_train_res,
        X_val_seq=to_pseudo_sequence(X_val),
        y_val=y_val,
        X_test_seq=to_pseudo_sequence(X_test),
        y_test=y_test,
        text_train=df["narration"].iloc[idx_train].reset_index(drop=True),
        y_train_orig=y_train,
        text_val=df["narration"].iloc[idx_val].reset_index(drop=True),
        text_test=df["narration"].iloc[idx_test].reset_index(drop=True),
        train_counts_before=before,
        train_counts_after=after,
    )
