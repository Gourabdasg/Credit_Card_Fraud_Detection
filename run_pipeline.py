"""End-to-end Review-3 pipeline on the synthetic proxy dataset.

Run:  python run_pipeline.py
Outputs (in ./outputs): metrics_table.csv, run_summary.json, roc_curves.png, confusion_matrix.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from config import OUTPUT_DIR, SEED
from data_synthetic import make_proxy_dataset
from evaluate import metrics_row, metrics_table, plot_confusion, plot_roc
from fusion import StackedFusion, simple_average
from models import (
    build_cnn,
    build_lstm,
    build_nlp,
    set_global_seed,
    train_deep_model,
)
from preprocessing import prepare


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_global_seed(SEED)

    # 1. Data (synthetic proxy; real Kaggle data is used from Review-4)
    df = make_proxy_dataset()
    d = prepare(df)
    print(f"Samples: {len(df)} | test: {len(d.y_test)} ({int(d.y_test.sum())} fraud)")
    print(f"Train before SMOTE (genuine/fraud): {d.train_counts_before}")
    print(f"Train after  SMOTE (genuine/fraud): {d.train_counts_after}")

    # 2. Branches
    lstm = build_lstm()
    h_lstm = train_deep_model(lstm, d.X_train_seq, d.y_train, d.X_val_seq, d.y_val)
    cnn = build_cnn()
    h_cnn = train_deep_model(cnn, d.X_train_seq, d.y_train, d.X_val_seq, d.y_val)
    nlp = build_nlp().fit(d.text_train, d.y_train_orig)
    epochs = {"LSTM": len(h_lstm.history["loss"]), "CNN": len(h_cnn.history["loss"])}
    print(f"Epochs run (early stopping): {epochs}")

    def probs(split_seq, split_text):
        return (
            lstm.predict(split_seq, verbose=0).ravel(),
            cnn.predict(split_seq, verbose=0).ravel(),
            nlp.predict_proba(split_text)[:, 1],
        )

    val_l, val_c, val_n = probs(d.X_val_seq, d.text_val)
    te_l, te_c, te_n = probs(d.X_test_seq, d.text_test)

    # 3. Fusion
    stacked = StackedFusion(["LSTM", "CNN", "NLP"]).fit([val_l, val_c, val_n], d.y_val)
    te_stack = stacked.predict_proba([te_l, te_c, te_n])
    te_avg = simple_average([te_l, te_n])
    print(f"Stacked meta-weights: {stacked.weights} | tuned threshold: {stacked.threshold:.2f}")

    # 4. Evaluation on the test split
    rows = {
        "LSTM branch": metrics_row(d.y_test, te_l),
        "1D-CNN branch": metrics_row(d.y_test, te_c),
        "NLP branch (TF-IDF + LogReg)": metrics_row(d.y_test, te_n),
        "Simple-average fusion (LSTM + NLP)": metrics_row(d.y_test, te_avg),
        "Stacked fusion (LSTM + CNN + NLP)": metrics_row(d.y_test, te_stack, stacked.threshold),
    }
    table = metrics_table(rows)
    print("\n" + table.to_string())
    table.to_csv(OUTPUT_DIR / "metrics_table.csv")

    plot_roc(
        d.y_test,
        {
            "LSTM": te_l,
            "1D-CNN": te_c,
            "NLP": te_n,
            "Stacked fusion": te_stack,
        },
        OUTPUT_DIR / "roc_curves.png",
    )
    plot_confusion(
        d.y_test,
        stacked.predict([te_l, te_c, te_n]),
        OUTPUT_DIR / "confusion_matrix.png",
        "Stacked fusion (test, proxy data)",
    )

    summary = {
        "n_test": int(len(d.y_test)),
        "n_test_fraud": int(d.y_test.sum()),
        "train_counts_before_smote": d.train_counts_before,
        "train_counts_after_smote": d.train_counts_after,
        "epochs_run": epochs,
        "meta_weights": {k: float(v) for k, v in stacked.weights.items()},
        "tuned_threshold": stacked.threshold,
        "note": "Synthetic proxy data - pipeline-development results, not final results.",
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
