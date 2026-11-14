# Credit Card Fraud and Alert System using Deep Learning and NLP

Full project code (Reviews 1-6). Brainware University, B.Tech CSE (AI & ML).

## Setup

```bash
pip install -r requirements.txt        # Python 3.10-3.12 recommended for TensorFlow
```

## Three ways to run

| Command | What it does | Reviews |
|---|---|---|
| `python run_pipeline.py` | Synthetic proxy data (6,000 rows). LSTM + CNN + NLP, SMOTE, stacked fusion. | 1-3 |
| `python run_full.py --data data/creditcard.csv` | **Real Kaggle data.** Ordered transaction windows, training, stacked fusion, test metrics, saves models. | 4, 6 |
| `python simulate_alerts.py --data data/creditcard.csv` | Replays transactions through the real-time alert module (needs `run_full.py` first). | 5 |

Get the dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud -> put `creditcard.csv` in `data/`.

Quick test without the real file (NOT real data - never report these numbers):

```bash
python tools/make_demo_csv.py --out data/demo_creditcard.csv
python run_full.py --data data/demo_creditcard.csv --max-epochs 4
python simulate_alerts.py --data data/demo_creditcard.csv
```

Useful options for `run_full.py`: `--split chronological` (stricter, leakage-free split), `--window 5`,
`--max-epochs 40`, `--sample 30000` (debug only), `--verbose 1`.

## Project layout

```
run_pipeline.py          Review 1-3 pipeline on proxy data
run_full.py              Review 4/6: real Kaggle training + final evaluation
simulate_alerts.py       Review 5: real-time alert demo
src/config.py            all settings
src/data_synthetic.py    proxy dataset + synthetic narration text
src/data_kaggle.py       Kaggle loader (sorted by Time)
src/preprocessing.py     proxy-data prep (pseudo-sequence)
src/preprocessing_seq.py real-data prep (ordered windows, scaling, SMOTE on train only)
src/models.py            LSTM, 1D-CNN, TF-IDF + LogReg
src/fusion.py            simple average + stacked meta-classifier + threshold tuning
src/evaluate.py          precision / recall / F1 / AUC-ROC / PR-AUC, ROC, PR, confusion matrix
src/persist.py           save / load trained models
src/alert.py             FraudAlertSystem (rolling window, risk level, reasons, JSONL log)
tools/make_demo_csv.py   fake Kaggle-format file for testing
docs/                    text copies of the Review-1/2/3 reports
outputs/                 results (proxy run in outputs/, real run in outputs/full/)
```

## Things to state honestly in the final report

- **No cardholder ID in Kaggle data.** Sequences are windows of consecutive transactions ordered by `Time`, not per-cardholder histories.
- **No text in Kaggle data.** The NLP branch uses synthetic narration generated from the label (`make_narrations`), so its score is illustrative and optimistic. Replace with real text if you can find a source.
- **Split:** `stratified` matches the Review-1 plan, but overlapping windows can sit in different splits. Report the `chronological` result too.
- **Fraud is ~0.17%:** quote PR-AUC and recall/precision, not accuracy.
- Metrics from the proxy run will not match the Review-3 table exactly (the original generator/seed was not available).
