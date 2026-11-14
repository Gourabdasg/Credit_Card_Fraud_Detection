"""Create a small synthetic file in the Kaggle creditcard.csv format, for testing the full
pipeline without the real 150 MB dataset.

    python tools/make_demo_csv.py --out data/demo_creditcard.csv --rows 20000 --fraud 120

The result is NOT real data - never report numbers from it as Kaggle results.
"""

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_creditcard.csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--fraud", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_gen = args.rows - args.fraud
    gen = rng.normal(0, 1, (n_gen, 28))
    fra = rng.normal(0, 1.3, (args.fraud, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 13, 16]] = [-1.5, -1.3, 1.3, -1.2, -1.2, -1.0]
    fra += shift

    X = np.vstack([gen, fra])
    y = np.r_[np.zeros(n_gen, int), np.ones(args.fraud, int)]
    amount = np.r_[rng.lognormal(3.5, 1.0, n_gen), rng.lognormal(3.9, 1.4, args.fraud)]
    time_s = np.sort(rng.uniform(0, 172792, args.rows))

    perm = rng.permutation(args.rows)  # fraud scattered over time
    df = pd.DataFrame(X[perm], columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time_s)
    df["Amount"] = amount[perm]
    df["Class"] = y[perm]
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(df)} rows, {int(df['Class'].sum())} fraud")


if __name__ == "__main__":
    main()
