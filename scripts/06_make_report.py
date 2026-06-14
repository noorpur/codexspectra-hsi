#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from codexspectra.utils.config import ensure_dirs, load_config


def plot_confusion(cm_path: Path, out_path: Path) -> None:
    cm = pd.read_csv(cm_path, index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.iloc[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = Path(cfg["reporting"]["tables_dir"])
    figs = Path(cfg["reporting"]["figures_dir"])
    ensure_dirs(tables, figs)
    cm_path = tables / "confusion_matrix_test.csv"
    if cm_path.exists():
        plot_confusion(cm_path, figs / "confusion_matrix_test.png")
    metrics_path = tables / "metrics_test.csv"
    metrics_md = "Metrics not generated yet. Run scripts/05_evaluate.py first."
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).to_markdown(index=False)
        metrics_md = metrics
    report = f"""# Experiment report

## Configuration

- Config: `{cfg.get('_config_path')}`
- Config hash: `{cfg.get('_config_hash')}`

## Test metrics

{metrics_md}

## Figures

- `reports/figures/confusion_matrix_test.png`
- `reports/figures/spectral_means_by_material.png` (generated in full analysis run)
- `reports/figures/pca_latent_space.png` (generated in full analysis run)
- `reports/figures/wavelength_importance.png` (generated in full analysis run)
- `reports/figures/uncertainty_map_example.png` (generated in full analysis run)

## Interpretation checklist

- Do the top wavelengths align with known reflectance differences between ink/support/material regions?
- Are errors concentrated in mixed-material or degraded/aged regions?
- Are per-document metrics stable, or is the model overfitting acquisition-specific signatures?
- Does group-aware performance remain credible compared with pixel-random splits?
"""
    Path("reports/experiment_report.md").write_text(report, encoding="utf-8")
    print("Wrote reports/experiment_report.md")


if __name__ == "__main__":
    main()
