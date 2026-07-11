"""Locked evaluation metrics. PR_AUC is primary (imbalance-honest), ROC_AUC
secondary, F1 at a fixed threshold an operating-point check (tuned separately),
Brier the calibration tracker the LLM layer depends on. Accuracy is a cautionary
foil, never optimized."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, brier_score_loss, accuracy_score)


def evaluate(y_true, y_prob, threshold=0.5):
    """Return the locked metric dict for probabilistic predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "PR_AUC":            average_precision_score(y_true, y_prob),
        "ROC_AUC":           roc_auc_score(y_true, y_prob),
        f"F1@{threshold:g}": f1_score(y_true, y_pred, zero_division=0),
        "Brier":             brier_score_loss(y_true, y_prob),
        "Accuracy":          accuracy_score(y_true, y_pred),
    }


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion (k of n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)