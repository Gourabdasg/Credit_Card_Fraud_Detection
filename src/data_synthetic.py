"""Synthetic proxy dataset generator.

The full Kaggle Credit Card Fraud Detection dataset is scheduled for Review-4. Until then the
pipeline is developed on a synthetic proxy that mirrors its severe class imbalance and has the
same column layout (V1..V28, Amount, Class) plus a synthetic transaction-narration text column.

NOTE: the narration text is deliberately strongly linked to the label (see Review-3 limitations),
so NLP-branch scores on this proxy are optimistic compared with what real text would give.
"""

import numpy as np
import pandas as pd

from config import N_FRAUD, N_PCA_FEATURES, N_SAMPLES, SEED

GENUINE_TEMPLATES = [
    "grocery purchase at local supermarket",
    "monthly subscription payment streaming service",
    "fuel station payment",
    "restaurant dinner payment",
    "pharmacy purchase",
    "online bookstore order",
    "utility bill payment",
    "public transport ticket",
    "coffee shop purchase",
    "clothing store purchase",
]

FRAUD_TEMPLATES = [
    "urgent overseas wire transfer unknown merchant",
    "multiple rapid purchases foreign country new device",
    "high value electronics order shipping address mismatch",
    "gift card bulk purchase suspicious merchant",
    "cash advance atm abroad unusual time",
    "card not present payment unverified website",
    "test transaction small amount followed by large charge",
    "international transfer flagged login from new location",
]

# Neutral words that appear in either class, so the text is not 100% separable.
NOISE_WORDS = ["payment", "order", "purchase", "online", "store", "card", "transaction"]


def _narration(
    is_fraud: bool,
    rng: np.random.Generator,
    fraud_rate: float = 0.70,
    genuine_fp_rate: float = 0.0,
) -> str:
    # Most fraud rows use a fraud-style narration; a minority look like ordinary purchases.
    if is_fraud:
        use_fraud_text = rng.random() < fraud_rate
    elif genuine_fp_rate > 0:
        use_fraud_text = rng.random() < genuine_fp_rate
    else:
        use_fraud_text = False  # genuine rows never use fraud-style text (strong label link)
    base = rng.choice(FRAUD_TEMPLATES if use_fraud_text else GENUINE_TEMPLATES)
    extra = " ".join(rng.choice(NOISE_WORDS, size=rng.integers(0, 3)))
    return f"{base} {extra}".strip()


def make_proxy_dataset(seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with V1..V28, Amount, narration and Class (0 genuine / 1 fraud)."""
    rng = np.random.default_rng(seed)
    n_genuine = N_SAMPLES - N_FRAUD

    # Genuine: standard normal PCA-like features. Fraud: mean-shifted on a few features, wider spread.
    genuine = rng.normal(0.0, 1.0, size=(n_genuine, N_PCA_FEATURES))
    fraud = rng.normal(0.0, 1.3, size=(N_FRAUD, N_PCA_FEATURES))
    shift = np.zeros(N_PCA_FEATURES)
    shift[[0, 2, 3, 9, 13, 16]] = [-0.9, -0.8, 0.8, -0.7, -0.7, -0.6]
    fraud = fraud + shift

    amount_genuine = rng.lognormal(mean=3.5, sigma=1.0, size=n_genuine)
    amount_fraud = rng.lognormal(mean=3.9, sigma=1.4, size=N_FRAUD)

    X = np.vstack([genuine, fraud])
    amount = np.concatenate([amount_genuine, amount_fraud])
    y = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(N_FRAUD, dtype=int)])

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, N_PCA_FEATURES + 1)])
    df["Amount"] = amount
    df["narration"] = [_narration(bool(label), rng) for label in y]
    df["Class"] = y

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_narrations(
    labels,
    seed: int = SEED,
    fraud_rate: float = 0.70,
    genuine_fp_rate: float = 0.02,
) -> list:
    """Synthetic narration text for an existing label vector (used with the Kaggle data,
    which has no text column). Deterministic for a given (labels, seed)."""
    rng = np.random.default_rng(seed)
    return [
        _narration(bool(lab), rng, fraud_rate=fraud_rate, genuine_fp_rate=genuine_fp_rate)
        for lab in labels
    ]


if __name__ == "__main__":
    data = make_proxy_dataset()
    print(data.shape)
    print(data["Class"].value_counts())
    print(data.head())
