#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from codexspectra.evaluation.metrics import bootstrap_macro_f1, classification_metrics, confusion_frame, report_frame
from codexspectra.utils.config import ensure_dirs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="best_tuned_random_forest.joblib")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data, reporting = cfg["data"], cfg["reporting"]
    ensure_dirs(reporting["tables_dir"])
    df = pd.read_csv(data["feature_table_path"])
    feat = [c for c in df.columns if c.startswith("f")]
    test = df.split.eq("test")
    model_path = Path(reporting["model_dir"]) / args.model
    if not model_path.exists():
        model_path = Path(reporting["model_dir"]) / "best_baseline.joblib"
    model = joblib.load(model_path)
    y_true = df.loc[test, data["label_column"]].astype(str)
    y_pred = model.predict(df.loc[test, feat])
    labels = sorted(pd.Series(y_true).unique().tolist())
    metrics = {"model_path": str(model_path), **classification_metrics(y_true, y_pred), **bootstrap_macro_f1(y_true, y_pred)}
    pd.DataFrame([metrics]).to_csv(Path(reporting["tables_dir"]) / "metrics_test.csv", index=False)
    report_frame(y_true, y_pred).to_csv(Path(reporting["tables_dir"]) / "classification_report_test.csv")
    confusion_frame(y_true, y_pred, labels).to_csv(Path(reporting["tables_dir"]) / "confusion_matrix_test.csv")
    print(metrics)


if __name__ == "__main__":
    main()
