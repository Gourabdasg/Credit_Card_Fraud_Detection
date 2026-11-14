"""Full pipeline on the REAL Kaggle dataset (Reviews 4-6): train, tune fusion, evaluate, save models.

Usage
    python run_full.py --data data/creditcard.csv
    python run_full.py --data data/creditcard.csv --split chronological
    python run_full.py --data data/creditcard.csv --sample 30000 --max-epochs 5     # quick debug run

Outputs (outputs/full/): metrics_table.csv, run_summary.json, roc_curves.png, pr_curves.png,
confusion_matrix.png.  Trained models go to models/ (used by simulate_alerts.py / alert.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (
    FULL_OUTPUT_DIR,
    KAGGLE_BATCH_SIZE,
    KAGGLE_CSV,
    MAX_EPOCHS,
    MODEL_DIR,
    SEED,
    SEQ_WINDOW,
    SPLIT_MODE,
    SYNTH_TEXT_FRAUD_RATE,
    SYNTH_TEXT_GENUINE_FP_RATE,
)
from data_kaggle import describe, load_kaggle
from data_synthetic import make_narrations
from evaluate import metrics_row, metrics_table, plot_confusion, plot_pr, plot_roc
from fusion import StackedFusion, simple_average
from models import build_cnn, build_lstm, build_nlp, set_global_seed, train_deep_model
from persist import save_artifacts
from preprocessing_seq import prepare_sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(KAGGLE_CSV), help="path to creditcard.csv")
    ap.add_argument("--split", default=SPLIT_MODE, choices=["stratified", "chronological"])
    ap.add_argument("--window", type=int, default=SEQ_WINDOW)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--sample", type=int, default=None, help="debug: keep all fraud + N total rows")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    # 1. Data
    df = load_kaggle(args.data, sample=args.sample)
    info = describe(df)
    print("Dataset:", info)
    df["narration"] = make_narrations(
        df["Class"].to_numpy(),
        seed=SEED,
        fraud_rate=SYNTH_TEXT_FRAUD_RATE,
        genuine_fp_rate=SYNTH_TEXT_GENUINE_FP_RATE,
    )
    d = prepare_sequences(df, window=args.window, mode=args.split)
    print(f"Split mode: {d.split_mode} | windows: train(orig)={sum(d.train_counts_before)} "
          f"val={len(d.y_val)} test={len(d.y_test)} (test fraud={int(d.y_test.sum())})")
    print(f"Train before SMOTE (genuine/fraud): {d.train_counts_before}")
    print(f"Train after  SMOTE (genuine/fraud): {d.train_counts_after}")

    # 2. Branches
    fit_kw = dict(batch_size=KAGGLE_BATCH_SIZE, epochs=args.max_epochs, verbose=args.verbose)
    lstm = build_lstm(d.window, d.n_features)
    h_lstm = train_deep_model(lstm, d.X_train_seq, d.y_train, d.X_val_seq, d.y_val, **fit_kw)
    cnn = build_cnn(d.window, d.n_features)
    h_cnn = train_deep_model(cnn, d.X_train_seq, d.y_train, d.X_val_seq, d.y_val, **fit_kw)
    nlp = build_nlp().fit(d.text_train, d.y_train_orig)
    epochs = {"LSTM": len(h_lstm.history["loss"]), "CNN": len(h_cnn.history["loss"])}
    print("Epochs run (early stopping):", epochs)

    def probs(seq, text):
        return (
            lstm.predict(seq, batch_size=1024, verbose=0).ravel(),
            cnn.predict(seq, batch_size=1024, verbose=0).ravel(),
            nlp.predict_proba(text)[:, 1],
        )

    val_l, val_c, val_n = probs(d.X_val_seq, d.text_val)
    te_l, te_c, te_n = probs(d.X_test_seq, d.text_test)

    # 3. Fusion (meta-classifier + threshold fitted on the validation split)
    stacked = StackedFusion(["LSTM", "CNN", "NLP"]).fit([val_l, val_c, val_n], d.y_val)
    te_stack = stacked.predict_proba([te_l, te_c, te_n])
    te_avg = simple_average([te_l, te_n])
    print(f"Stacked meta-weights: {stacked.weights} | tuned threshold: {stacked.threshold:.2f}")

    # 4. Evaluation on the held-out test split
    rows = {
        "LSTM branch": metrics_row(d.y_test, te_l),
        "1D-CNN branch": metrics_row(d.y_test, te_c),
        "NLP branch (TF-IDF + LogReg)": metrics_row(d.y_test, te_n),
        "Simple-average fusion (LSTM + NLP)": metrics_row(d.y_test, te_avg),
        "Stacked fusion (LSTM + CNN + NLP)": metrics_row(d.y_test, te_stack, stacked.threshold),
    }
    table = metrics_table(rows)
    print("\n" + table.to_string())
    table.to_csv(FULL_OUTPUT_DIR / "metrics_table.csv")

    curves = {"LSTM": te_l, "1D-CNN": te_c, "NLP": te_n, "Stacked fusion": te_stack}
    plot_roc(d.y_test, curves, FULL_OUTPUT_DIR / "roc_curves.png")
    plot_pr(d.y_test, curves, FULL_OUTPUT_DIR / "pr_curves.png")
    plot_confusion(d.y_test, stacked.predict([te_l, te_c, te_n]),
                   FULL_OUTPUT_DIR / "confusion_matrix.png", "Stacked fusion (test, Kaggle data)")

    summary = {
        "dataset": info,
        "split_mode": d.split_mode,
        "window": d.window,
        "n_test": int(len(d.y_test)),
        "n_test_fraud": int(d.y_test.sum()),
        "train_counts_before_smote": d.train_counts_before,
        "train_counts_after_smote": d.train_counts_after,
        "epochs_run": epochs,
        "meta_weights": {k: float(v) for k, v in stacked.weights.items()},
        "tuned_threshold": stacked.threshold,
        "debug_sample": args.sample,
        "note": "NLP text is synthetic (Kaggle data has no narration column).",
    }
    (FULL_OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # 5. Save models for the alert module
    save_artifacts(MODEL_DIR, lstm, cnn, nlp, stacked, d.scaler, d.window, d.n_features,
                   extra={"split_mode": d.split_mode, "debug_sample": args.sample})
    print(f"\nModels saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
