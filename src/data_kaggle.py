"""Loader for the real Kaggle 'Credit Card Fraud Detection' dataset (Review-4).

Download creditcard.csv from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it at  data/creditcard.csv  (or pass --data <path> to run_full.py).

The file has: Time, V1..V28, Amount, Class. It has NO cardholder ID and NO text, so:
  * sequences are windows of consecutive transactions ordered by Time;
  * narration text is synthetic (see data_synthetic.make_narrations) and must be described as
    illustrative in the report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED

V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time", *V_COLS, "Amount", "Class"]


def load_kaggle(path, sample: int = None, seed: int = SEED) -> pd.DataFrame:
    """Read the CSV, validate columns, sort by Time.

    sample: debug option. Keeps ALL fraud rows and randomly keeps genuine rows until `sample`
    rows remain, then re-sorts by Time. Not for final results (it changes the fraud rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud) and put it in the data/ folder."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[REQUIRED].dropna().drop_duplicates().reset_index(drop=True)

    if sample is not None and sample < len(df):
        fraud = df[df["Class"] == 1]
        genuine = df[df["Class"] == 0]
        n_gen = max(sample - len(fraud), 0)
        genuine = genuine.sample(n=min(n_gen, len(genuine)), random_state=seed)
        df = pd.concat([fraud, genuine])

    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    return df


def describe(df: pd.DataFrame) -> dict:
    n_fraud = int(df["Class"].sum())
    return {
        "rows": int(len(df)),
        "fraud": n_fraud,
        "genuine": int(len(df) - n_fraud),
        "fraud_pct": round(100.0 * n_fraud / len(df), 4),
    }
