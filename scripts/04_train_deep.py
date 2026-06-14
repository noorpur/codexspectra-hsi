#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from codexspectra.training.deep_train import train_mlp
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
    train, val = df.split.eq("train"), df.split.eq("val")
    model, label_encoder, val_loss = train_mlp(
        df.loc[train, feat].to_numpy("float32"),
        df.loc[train, data["label_column"]].astype(str).to_numpy(),
        df.loc[val, feat].to_numpy("float32"),
        df.loc[val, data["label_column"]].astype(str).to_numpy(),
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        seed=cfg["project"]["seed"],
        device=cfg["training"]["device"],
    )
    torch.save(model.state_dict(), Path(reporting["model_dir"]) / "spectral_mlp.pt")
    joblib.dump(label_encoder, Path(reporting["model_dir"]) / "spectral_mlp_label_encoder.joblib")
    pd.DataFrame([{"model": "spectral_mlp", "val_loss": val_loss}]).to_csv(
        Path(reporting["tables_dir"]) / "deep_validation_metrics.csv", index=False
    )
    print(f"Saved spectral MLP with validation loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
