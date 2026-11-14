"""Real-time fraud alert module (Review-5).

FraudAlertSystem.process(tx) takes ONE incoming transaction (a dict with Time, V1..V28, Amount
and optionally narration), keeps a rolling window of the most recent transactions, scores it with
the LSTM + CNN + NLP branches fused by the stacked meta-classifier, and raises an alert when the
fused score reaches the validation-tuned threshold.

Alerts are (a) passed to every notifier (console by default), and (b) appended to a JSONL log.
Add your own notifier (email / SMS / webhook) by passing any callable that accepts the alert dict.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

from config import ALERT_LOG, MODEL_DIR
from persist import load_artifacts
from preprocessing_seq import BASE_COLS


def console_notifier(alert: dict) -> None:
    print(
        f"[FRAUD ALERT] {alert['risk']:<8} id={alert['transaction_id']} "
        f"amount={alert['amount']:.2f} score={alert['score']:.3f}\n"
        f"              {alert['message']}"
    )


class FraudAlertSystem:
    def __init__(self, model_dir=MODEL_DIR, log_path=ALERT_LOG, notifiers=None):
        art = load_artifacts(model_dir)
        self.lstm, self.cnn, self.nlp = art["lstm"], art["cnn"], art["nlp"]
        self.fusion, self.scaler, self.meta = art["fusion"], art["scaler"], art["meta"]
        self.window = int(self.meta["window"])
        self.threshold = float(self.meta["threshold"])
        self.notifiers = [console_notifier] if notifiers is None else list(notifiers)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=self.window)
        self.prev_time = None
        self.count = 0

    # ------------------------------------------------------------------
    def _risk(self, score: float) -> str:
        if score < self.threshold:
            return "OK"
        midpoint = self.threshold + (1.0 - self.threshold) / 2.0
        return "CRITICAL" if score >= midpoint else "HIGH"

    def _reasons(self, p_lstm: float, p_cnn: float, p_nlp: float) -> list:
        reasons = []
        if p_nlp >= 0.5:
            reasons.append(f"suspicious transaction text (NLP score {p_nlp:.2f})")
        if p_lstm >= 0.5:
            reasons.append(f"unusual transaction sequence (LSTM score {p_lstm:.2f})")
        if p_cnn >= 0.5:
            reasons.append(f"unusual transaction pattern (CNN score {p_cnn:.2f})")
        return reasons or ["combined evidence from all models"]

    # ------------------------------------------------------------------
    def process(self, tx: dict) -> dict:
        """Score one transaction. Returns a result dict; result['alert'] is True if flagged."""
        self.count += 1
        t = float(tx["Time"])
        dt = 0.0 if self.prev_time is None else max(t - self.prev_time, 0.0)
        self.prev_time = t

        row = [float(tx[c]) for c in BASE_COLS] + [float(np.log1p(dt))]
        self.buffer.append(np.asarray(row, dtype=np.float64))

        rows = list(self.buffer)
        while len(rows) < self.window:           # warm-up: repeat the oldest row
            rows.insert(0, rows[0])
        w = self.scaler.transform(np.vstack(rows)).astype(np.float32)[None, :, :]

        p_lstm = float(self.lstm(w, training=False).numpy().ravel()[0])
        p_cnn = float(self.cnn(w, training=False).numpy().ravel()[0])
        p_nlp = float(self.nlp.predict_proba([str(tx.get("narration", ""))])[0, 1])
        score = float(
            self.fusion.predict_proba([np.array([p_lstm]), np.array([p_cnn]), np.array([p_nlp])])[0]
        )
        risk = self._risk(score)
        is_alert = risk != "OK"

        result = {
            "transaction_id": tx.get("id", self.count),
            "time": t,
            "amount": float(tx["Amount"]),
            "narration": tx.get("narration", ""),
            "score": score,
            "threshold": self.threshold,
            "branch_scores": {"lstm": p_lstm, "cnn": p_cnn, "nlp": p_nlp},
            "risk": risk,
            "alert": is_alert,
        }
        if is_alert:
            result["message"] = (
                f"Possible fraud on transaction {result['transaction_id']} "
                f"(amount {result['amount']:.2f}): " + "; ".join(self._reasons(p_lstm, p_cnn, p_nlp))
            )
            self._dispatch(result)
        return result

    def _dispatch(self, alert: dict) -> None:
        for notify in self.notifiers:
            notify(alert)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert) + "\n")
