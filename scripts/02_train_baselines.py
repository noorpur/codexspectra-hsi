#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from codexspectra.evaluation.metrics import classification_metrics, report_frame
from codexspectra.models.baselines import baseline_models
from codexspectra.utils.config import ensure_dirs, load_config
from codexspectra.utils.repro import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    data, reporting = cfg["data"], cfg["reporting"]
    ensure_dirs(reporting["model_dir"], reporting["tables_dir"])
    df = pd.read_csv(data["feature_table_path"])
    feat = [c for c in df.columns if c.startswith("f")]
    train = df.split.eq("train")
    val = df.split.eq("val")
    models = baseline_models(cfg["project"]["seed"], cfg["training"]["n_jobs"])
    rows = []
    best_name, best_score, best_model = None, -1.0, None
    for name, model in models.items():
        model.fit(df.loc[train, feat], df.loc[train, data["label_column"]])
        pred = model.predict(df.loc[val, feat])
        metrics = classification_metrics(df.loc[val, data["label_column"]], pred)
        rows.append({"model": name, **metrics})
        joblib.dump(model, Path(reporting["model_dir"]) / f"{name}.joblib")
        if metrics[cfg["training"]["metric"]] > best_score:
            best_name, best_score, best_model = name, metrics[cfg["training"]["metric"]], model
    pd.DataFrame(rows).sort_values(cfg["training"]["metric"], ascending=False).to_csv(
        Path(reporting["tables_dir"]) / "baseline_validation_metrics.csv", index=False
    )
    joblib.dump(best_model, Path(reporting["model_dir"]) / "best_baseline.joblib")
    print(f"Best validation model: {best_name} ({cfg['training']['metric']}={best_score:.4f})")


if __name__ == "__main__":
    main()
