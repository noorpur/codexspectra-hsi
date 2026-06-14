#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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
    xtr, ytr = df.loc[train, feat], df.loc[train, data["label_column"]]
    xva, yva = df.loc[val, feat], df.loc[val, data["label_column"]]

    try:
        import optuna
    except ModuleNotFoundError:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["project"]["seed"],
        )
        model.fit(pd.concat([xtr, xva]), pd.concat([ytr, yva]))
        joblib.dump(model, Path(reporting["model_dir"]) / "best_tuned_random_forest.joblib")
        pd.DataFrame([{"note": "optuna_not_installed_fallback", "macro_f1": None}]).to_csv(
            Path(reporting["tables_dir"]) / "optuna_trials.csv", index=False
        )
        print("Optuna is not installed; saved deterministic fallback tuned Random Forest.")
        return

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced_subsample",
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["project"]["seed"],
        )
        model.fit(xtr, ytr)
        pred = model.predict(xva)
        return f1_score(yva, pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["training"]["optuna_trials"])
    best = RandomForestClassifier(
        **study.best_params,
        class_weight="balanced_subsample",
        n_jobs=cfg["training"]["n_jobs"],
        random_state=cfg["project"]["seed"],
    )
    best.fit(pd.concat([xtr, xva]), pd.concat([ytr, yva]))
    joblib.dump(best, Path(reporting["model_dir"]) / "best_tuned_random_forest.joblib")
    study.trials_dataframe().to_csv(Path(reporting["tables_dir"]) / "optuna_trials.csv", index=False)
    print(f"Best tuned RF macro F1: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
