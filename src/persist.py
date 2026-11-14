"""Save / load trained artifacts so the alert module can score new transactions."""

import json
from pathlib import Path

import joblib
from tensorflow import keras


def save_artifacts(model_dir, lstm, cnn, nlp, fusion, scaler, window: int, n_features: int,
                   extra: dict = None) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lstm.save(model_dir / "lstm.keras")
    cnn.save(model_dir / "cnn.keras")
    joblib.dump(nlp, model_dir / "nlp.joblib")
    joblib.dump(fusion, model_dir / "fusion.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    meta = {
        "window": window,
        "n_features": n_features,
        "threshold": float(fusion.threshold),
        "meta_weights": {k: float(v) for k, v in fusion.weights.items()},
    }
    meta.update(extra or {})
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(model_dir) -> dict:
    model_dir = Path(model_dir)
    if not (model_dir / "meta.json").exists():
        raise FileNotFoundError(
            f"No trained models in {model_dir}. Run  python run_full.py --data <creditcard.csv>  first."
        )
    return {
        "lstm": keras.models.load_model(model_dir / "lstm.keras"),
        "cnn": keras.models.load_model(model_dir / "cnn.keras"),
        "nlp": joblib.load(model_dir / "nlp.joblib"),
        "fusion": joblib.load(model_dir / "fusion.joblib"),
        "scaler": joblib.load(model_dir / "scaler.joblib"),
        "meta": json.loads((model_dir / "meta.json").read_text()),
    }
