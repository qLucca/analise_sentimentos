from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
