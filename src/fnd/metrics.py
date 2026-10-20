from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             precision_score, recall_score)


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y_true = np.asarray(y_true); p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(total)


def reliability(y_true, p, n_bins: int = 10):
    y_true = np.asarray(y_true); p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "mean_pred": float(p[m].mean()),
                        "frac_pos": float(y_true[m].mean()), "count": int(m.sum())})
    return out


def scores(y_true, p, threshold: float = 0.5) -> dict:
    yhat = (np.asarray(p) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "macro_f1": float(f1_score(y_true, yhat, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": ece(y_true, p),
    }
