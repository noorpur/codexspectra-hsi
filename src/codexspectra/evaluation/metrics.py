from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def report_frame(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T


def confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def bootstrap_macro_f1(y_true, y_pred, n: int = 500, seed: int = 1337) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_ci_low": float(np.percentile(scores, 2.5)),
        "macro_f1_ci_high": float(np.percentile(scores, 97.5)),
    }
