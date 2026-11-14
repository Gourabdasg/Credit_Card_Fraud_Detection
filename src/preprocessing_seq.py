"""Real-data preparation (Review-4): ordered transaction windows replace the pseudo-sequence.

Steps
  1. Sort by Time; per-row features = V1..V28, Amount, log1p(seconds since previous transaction).
  2. Each sample is a window of the last SEQ_WINDOW transactions; its label / text are those of
     the window's last transaction.
  3. Split windows 70:15:15 (stratified on the fraud label, or chronological).
  4. Standard scaling fitted on the training rows only.
  5. SMOTE on the training split only (flattened windows), so nothing synthetic reaches val/test.

Note: with a stratified split, neighbouring windows share transactions and can land in different
splits. Use SPLIT_MODE = "chronological" for a stricter, leakage-free (but harder) evaluation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from smote_compat import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SEED, SEQ_WINDOW, SMOTE_STRATEGY, SPLIT_MODE, TEST_FRAC, VAL_FRAC

V_COLS = [f"V{i}" for i in range(1, 29)]
BASE_COLS = V_COLS + ["Amount"]
FEATURE_NAMES = BASE_COLS + ["log_dt"]
N_FEATURES = len(FEATURE_NAMES)  # 30


@dataclass
class PreparedSeq:
    X_train_seq: np.ndarray        # after SMOTE, (n, window, 30)
    y_train: np.ndarray            # after SMOTE
    X_val_seq: np.ndarray
    y_val: np.ndarray
    X_test_seq: np.ndarray
    y_test: np.ndarray
    text_train: pd.Series          # original, un-resampled training split
    y_train_orig: np.ndarray
    text_val: pd.Series
    text_test: pd.Series
    train_counts_before: tuple
    train_counts_after: tuple
    scaler: StandardScaler
    window: int
    n_features: int
    split_mode: str


def row_features(df: pd.DataFrame) -> np.ndarray:
    """(n, 30) raw feature matrix for a Time-sorted frame."""
    t = df["Time"].to_numpy(dtype=np.float64)
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    return np.column_stack([df[BASE_COLS].to_numpy(dtype=np.float64), np.log1p(dt)])


def split_indices(y: np.ndarray, mode: str, seed: int = SEED):
    n = len(y)
    idx = np.arange(n)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round(n * VAL_FRAC))
    if mode == "chronological":
        n_train = n - n_val - n_test
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    if mode != "stratified":
        raise ValueError("split mode must be 'stratified' or 'chronological'")
    idx_train, idx_tmp = train_test_split(
        idx, test_size=n_val + n_test, stratify=y, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=n_test, stratify=y[idx_tmp], random_state=seed
    )
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def prepare_sequences(
    df: pd.DataFrame,
    window: int = SEQ_WINDOW,
    mode: str = SPLIT_MODE,
    seed: int = SEED,
) -> PreparedSeq:
    """df must be sorted by Time and have V1..V28, Amount, Time, Class, narration."""
    F = row_features(df)
    y_all = df["Class"].to_numpy()[window - 1 :]
    text_all = df["narration"].iloc[window - 1 :].reset_index(drop=True)

    idx_tr, idx_va, idx_te = split_indices(y_all, mode, seed)

    # Scaler fitted on the rows that END a training window (train statistics only).
    scaler = StandardScaler().fit(F[idx_tr + window - 1])
    Fs = scaler.transform(F).astype(np.float32)

    # windows[k] covers rows k .. k+window-1  ->  shape (n - window + 1, window, 30)
    windows = np.ascontiguousarray(
        sliding_window_view(Fs, window_shape=window, axis=0).transpose(0, 2, 1)
    )

    y_train, y_val, y_test = y_all[idx_tr], y_all[idx_va], y_all[idx_te]
    before = (int((y_train == 0).sum()), int((y_train == 1).sum()))

    flat = windows[idx_tr].reshape(len(idx_tr), -1)
    flat_res, y_res = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=seed).fit_resample(
        flat, y_train
    )
    X_train = flat_res.reshape(-1, window, F.shape[1]).astype(np.float32)
    after = (int((y_res == 0).sum()), int((y_res == 1).sum()))

    return PreparedSeq(
        X_train_seq=X_train,
        y_train=y_res,
        X_val_seq=windows[idx_va],
        y_val=y_val,
        X_test_seq=windows[idx_te],
        y_test=y_test,
        text_train=text_all.iloc[idx_tr].reset_index(drop=True),
        y_train_orig=y_train,
        text_val=text_all.iloc[idx_va].reset_index(drop=True),
        text_test=text_all.iloc[idx_te].reset_index(drop=True),
        train_counts_before=before,
        train_counts_after=after,
        scaler=scaler,
        window=window,
        n_features=F.shape[1],
        split_mode=mode,
    )
