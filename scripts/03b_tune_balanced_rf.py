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


def balanced_sample(df: pd.DataFrame, label_col: str, seed: int, max_per_class: int | None = None) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    n = counts.min()
    if max_per_class is not None:
        n = min(n, max_per_class)

    parts = []
    for label, group in df.groupby(label_col):
        parts.append(group.sample(n=n, random_state=seed, replace=False))

    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    data, reporting = cfg["data"], cfg["reporting"]
    ensure_dirs(reporting["model_dir"], reporting["tables_dir"])

    df = pd.read_csv(data["feature_table_path"])
    label_col = data["label_column"]
    feat = [c for c in df.columns if c.startswith("f")]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    train_val_df = df[df["split"].isin(["train", "val"])].copy()

    train_bal = balanced_sample(train_df, label_col, cfg["project"]["seed"], args.max_per_class)
    train_val_bal = balanced_sample(train_val_df, label_col, cfg["project"]["seed"], args.max_per_class)

    summary = pd.DataFrame({
        "original_train": train_df[label_col].value_counts(),
        "balanced_train": train_bal[label_col].value_counts(),
        "original_train_val": train_val_df[label_col].value_counts(),
        "balanced_train_val": train_val_bal[label_col].value_counts(),
    }).fillna(0).astype(int)
    summary.to_csv(Path(reporting["tables_dir"]) / "balanced_training_sample_summary.csv")

    xtr, ytr = train_bal[feat], train_bal[label_col]
    xva, yva = val_df[feat], val_df[label_col]

    try:
        import optuna
    except ModuleNotFoundError:
        optuna = None

    if optuna is None:
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["project"]["seed"],
        )
        model.fit(xtr, ytr)
        joblib.dump(model, Path(reporting["model_dir"]) / "best_balanced_random_forest_train_only.joblib")

        final = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["project"]["seed"],
        )
        final.fit(train_val_bal[feat], train_val_bal[label_col])
        joblib.dump(final, Path(reporting["model_dir"]) / "best_balanced_random_forest.joblib")
        print("Saved balanced fallback RF models.")
        return

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced_subsample",
            n_jobs=cfg["training"]["n_jobs"],
            random_state=cfg["project"]["seed"],
        )
        model.fit(xtr, ytr)
        pred = model.predict(xva)
        return f1_score(yva, pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["training"].get("optuna_trials", 20))

    best_train_only = RandomForestClassifier(
        **study.best_params,
        class_weight="balanced_subsample",
        n_jobs=cfg["training"]["n_jobs"],
        random_state=cfg["project"]["seed"],
    )
    best_train_only.fit(xtr, ytr)
    joblib.dump(best_train_only, Path(reporting["model_dir"]) / "best_balanced_random_forest_train_only.joblib")

    final = RandomForestClassifier(
        **study.best_params,
        class_weight="balanced_subsample",
        n_jobs=cfg["training"]["n_jobs"],
        random_state=cfg["project"]["seed"],
    )
    final.fit(train_val_bal[feat], train_val_bal[label_col])
    joblib.dump(final, Path(reporting["model_dir"]) / "best_balanced_random_forest.joblib")

    study.trials_dataframe().to_csv(Path(reporting["tables_dir"]) / "balanced_optuna_trials.csv", index=False)

    print("Best balanced RF validation macro-F1:", round(study.best_value, 4))
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
