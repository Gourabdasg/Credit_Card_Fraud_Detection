"""Evaluation helpers: metrics table, ROC curves and confusion matrix figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metrics_row(y_true, scores, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1-score": f1_score(y_true, pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, scores),
        # PR-AUC (average precision) is the more informative headline metric at ~0.17% fraud.
        "PR-AUC": average_precision_score(y_true, scores),
    }


def metrics_table(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).T.round(3)


def plot_roc(y_true, score_dict: dict, path) -> None:
    plt.figure(figsize=(6, 5))
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on test split (proxy data)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr(y_true, score_dict: dict, path, title: str = "Precision-recall curves (test split)") -> None:
    plt.figure(figsize=(6, 5))
    for name, scores in score_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
