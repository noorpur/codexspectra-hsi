#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize

from codexspectra.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="best_balanced_random_forest_train_only.joblib")
    ap.add_argument("--prefix", default="balanced")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data, reporting = cfg["data"], cfg["reporting"]

    df = pd.read_csv(data["feature_table_path"])
    feat = [c for c in df.columns if c.startswith("f")]
    label_col = data["label_column"]

    model_path = Path(reporting["model_dir"]) / args.model
    model = joblib.load(model_path)

    figures_dir = Path(reporting["figures_dir"])
    tables_dir = Path(reporting["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        pred = model.predict(sdf[feat])
        rows.append({
            "split": split,
            "n": len(sdf),
            "accuracy": accuracy_score(sdf[label_col], pred),
            "macro_f1": f1_score(sdf[label_col], pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(sdf[label_col], pred, average="weighted", zero_division=0),
        })

    perf = pd.DataFrame(rows)
    perf.to_csv(tables_dir / f"{args.prefix}_honest_overfitting_diagnostic.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(perf["split"], perf["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(perf["split"], perf["weighted_f1"], marker="o", label="Weighted-F1")
    plt.ylim(0, 1.05)
    plt.title("Honest overfitting diagnostic by split")
    plt.xlabel("Split")
    plt.ylabel("F1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.prefix}_honest_overfitting_diagnostic.png", dpi=220)
    plt.close()

    test = df[df["split"] == "test"]
    y_true = test[label_col].astype(str)
    y_pred = model.predict(test[feat])
    labels = sorted(y_true.unique())

    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    report.to_csv(tables_dir / f"{args.prefix}_classification_report_test.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = normalize(cm, norm="l1", axis=1)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(tables_dir / f"{args.prefix}_confusion_matrix_test_counts.csv")
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(tables_dir / f"{args.prefix}_confusion_matrix_test_normalized.csv")

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, aspect="auto")
    plt.title("Balanced model normalized test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="Row-normalized proportion")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.prefix}_confusion_matrix_test_normalized.png", dpi=220)
    plt.close()

    print(perf)
    print()
    print(report[["precision", "recall", "f1-score", "support"]])


if __name__ == "__main__":
    main()
