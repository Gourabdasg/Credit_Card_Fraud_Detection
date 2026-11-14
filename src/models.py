"""Branch models: LSTM, 1D-CNN (Keras) and TF-IDF + logistic regression (NLP)."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    CNN_FILTERS,
    CNN_KERNEL,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_EPOCHS,
    N_STEP_FEATURES,
    N_TIMESTEPS,
    SEED,
)


def build_lstm(n_timesteps: int = N_TIMESTEPS, n_features: int = N_STEP_FEATURES) -> keras.Model:
    """LSTM(32) -> Dropout(0.3) -> Dense(16, ReLU) -> Dense(1, sigmoid)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(n_timesteps, n_features)),
            layers.LSTM(LSTM_UNITS),
            layers.Dropout(DROPOUT),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_branch",
    )
    return model


def build_cnn(n_timesteps: int = N_TIMESTEPS, n_features: int = N_STEP_FEATURES) -> keras.Model:
    """Conv1D(32, k=3) -> GlobalMaxPool -> Dropout(0.3) -> Dense(16, ReLU) -> Dense(1, sigmoid)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(n_timesteps, n_features)),
            layers.Conv1D(CNN_FILTERS, CNN_KERNEL, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(DROPOUT),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_branch",
    )
    return model


def train_deep_model(
    model: keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose: int = 0,
    batch_size: int = BATCH_SIZE,
    epochs: int = MAX_EPOCHS,
):
    """Adam(1e-3), binary cross-entropy, batch 64, up to 40 epochs, early stopping (patience 5)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history


def set_global_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def build_nlp() -> Pipeline:
    """TF-IDF vectorisation + class-weighted logistic regression on narration text."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
