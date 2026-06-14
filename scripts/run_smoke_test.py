#!/usr/bin/env python
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hyperdoc"


def make_cube(label: str, group: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bands = 64
    x = np.linspace(0, 1, bands)
    if label == "ink":
        base = 0.30 + 0.08 * np.sin(8 * x) - 0.10 * np.exp(-((x - 0.35) ** 2) / 0.01)
    elif label == "substrate":
        base = 0.68 + 0.05 * x + 0.03 * np.cos(4 * x)
    else:
        base = 0.48 + 0.10 * np.exp(-((x - 0.65) ** 2) / 0.015)
    cube = base[None, None, :] + 0.025 * rng.normal(size=(32, 32, bands))
    out_dir = RAW / group / label
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{group}_{label}.npy", cube.astype("float32"))


def main() -> None:
    for group_i in range(8):
        for label_i, label in enumerate(["ink", "substrate", "aged"]):
            make_cube(label, f"doc{group_i:02d}", 1000 + group_i * 10 + label_i)
    commands = [
        ["python", "scripts/00_make_manifest.py", "--config", "configs/smoke.yaml"],
        ["python", "scripts/01_preprocess.py", "--config", "configs/smoke.yaml"],
        ["python", "scripts/02_train_baselines.py", "--config", "configs/smoke.yaml"],
        ["python", "scripts/03_tune_models.py", "--config", "configs/smoke.yaml"],
        ["python", "scripts/05_evaluate.py", "--config", "configs/smoke.yaml"],
        ["python", "scripts/06_make_report.py", "--config", "configs/smoke.yaml"],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Smoke test completed. Replace synthetic cubes with HYPERDOC raw data for real experiments.")


if __name__ == "__main__":
    main()
