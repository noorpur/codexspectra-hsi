#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from codexspectra.data.sampling import build_feature_table
from codexspectra.evaluation.splits import make_splits
from codexspectra.features.spectral import SpectralPreprocessor
from codexspectra.utils.config import ensure_dirs, load_config
from codexspectra.utils.repro import set_seed, write_run_metadata


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("b")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    data = cfg["data"]
    reporting = cfg["reporting"]
    ensure_dirs(data["processed_root"], reporting["model_dir"], reporting["tables_dir"])
    manifest = pd.read_csv(data["manifest_path"])
    raw = build_feature_table(manifest, data["max_pixels_per_cube"], cfg["project"]["seed"])
    if raw.empty:
        raise SystemExit("No spectra extracted. Check dataset paths and cube formats.")
    raw["split"] = make_splits(
        raw,
        label_col=data["label_column"],
        group_col=data["group_column"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=cfg["split"]["split_seed"],
        grouped=cfg["split"]["grouped"],
    )
    cols = feature_columns(raw)
    spec_cfg = {k: v for k, v in cfg["spectral"].items() if k in {"smoothing_window", "smoothing_polyorder", "use_snv", "use_derivatives", "pca_components"}}
    prep = SpectralPreprocessor(**spec_cfg)
    train_mask = raw["split"].eq("train")
    x_train = prep.fit_transform(raw.loc[train_mask, cols].to_numpy())
    x_other = prep.transform(raw.loc[~train_mask, cols].to_numpy())
    feat_cols = [f"f{i:03d}" for i in range(x_train.shape[1])]
    processed = raw.drop(columns=cols).copy()
    processed.loc[train_mask, feat_cols] = x_train
    processed.loc[~train_mask, feat_cols] = x_other
    out = Path(data["feature_table_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out, index=False)
    joblib.dump(prep, Path(reporting["model_dir"]) / "preprocessing_pipeline.joblib")
    processed.groupby(["split", data["label_column"]]).size().rename("n").reset_index().to_csv(
        Path(reporting["tables_dir"]) / "split_summary.csv", index=False
    )
    write_run_metadata(Path(reporting["tables_dir"]) / "run_metadata.json", cfg, {"n_rows": len(processed)})
    print(f"Wrote feature table to {out}")


if __name__ == "__main__":
    main()
