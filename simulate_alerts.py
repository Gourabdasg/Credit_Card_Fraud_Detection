"""Replay transactions through the real-time alert module (demo for Review-5 / Review-6).

Usage
    python simulate_alerts.py --data data/creditcard.csv            # demo around first fraud
    python simulate_alerts.py --data data/creditcard.csv --n 500 --start 120000

Requires trained models: run  python run_full.py --data ...  first.
NOTE: the replayed rows may come from the training split, so this demonstrates the alert flow,
not generalisation. Use the metrics from run_full.py for performance claims.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alert import FraudAlertSystem
from config import KAGGLE_CSV, SEED, SYNTH_TEXT_FRAUD_RATE, SYNTH_TEXT_GENUINE_FP_RATE
from data_kaggle import load_kaggle
from data_synthetic import make_narrations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(KAGGLE_CSV))
    ap.add_argument("--sample", type=int, default=None, help="must match the run_full.py --sample")
    ap.add_argument("--n", type=int, default=200, help="number of transactions to replay")
    ap.add_argument("--start", type=int, default=None, help="start row (default: 30 rows before first fraud)")
    args = ap.parse_args()

    df = load_kaggle(args.data, sample=args.sample)
    df["narration"] = make_narrations(
        df["Class"].to_numpy(), seed=SEED,
        fraud_rate=SYNTH_TEXT_FRAUD_RATE, genuine_fp_rate=SYNTH_TEXT_GENUINE_FP_RATE,
    )
    if args.start is None:
        first_fraud = int(df.index[df["Class"] == 1][0])
        args.start = max(first_fraud - 30, 0)
    chunk = df.iloc[args.start : args.start + args.n]

    system = FraudAlertSystem()
    t0 = time.time()
    n_alerts = tp = fp = fn = 0
    for rid, row in chunk.iterrows():
        tx = row.to_dict()
        tx["id"] = int(rid)
        res = system.process(tx)
        flagged, actual = res["alert"], int(row["Class"]) == 1
        n_alerts += flagged
        tp += flagged and actual
        fp += flagged and not actual
        fn += (not flagged) and actual

    dur = time.time() - t0
    print(f"\nReplayed {len(chunk)} transactions in {dur:.1f}s "
          f"({1000 * dur / max(len(chunk), 1):.1f} ms/transaction)")
    print(f"Alerts raised: {n_alerts} | caught fraud: {tp} | false alarms: {fp} | missed fraud: {fn}")
    print(f"Alert log: {system.log_path}")


if __name__ == "__main__":
    main()
