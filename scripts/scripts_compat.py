from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_label_table(path: Path) -> dict[int, dict[str, str]]:
    raw = pd.read_excel(path, sheet_name=0, engine="odf", header=None)
    mapping: dict[int, dict[str, str]] = {}

    for idx, row in raw.iterrows():
        if idx == 0:
            continue

        category = row.iloc[0]
        class_name = row.iloc[1] if len(row) > 1 else None
        subtype = row.iloc[2] if len(row) > 2 else None

        if pd.isna(category) and pd.isna(class_name):
            continue

        mapping[int(idx)] = {
            "category": "" if pd.isna(category) else str(category),
            "class_name": "" if pd.isna(class_name) else str(class_name),
            "subtype": "" if pd.isna(subtype) else str(subtype),
        }

    return mapping


def cube_to_hwc(cube: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    h, w = mask_shape

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    if cube.shape[1] == w and cube.shape[2] == h:
        return np.transpose(cube, (2, 1, 0))

    if cube.shape[1] == h and cube.shape[2] == w:
        return np.transpose(cube, (1, 2, 0))

    if cube.shape[0] == h and cube.shape[1] == w:
        return cube

    raise ValueError(f"Cannot align cube shape {cube.shape} with mask shape {mask_shape}")


def spectrum_features(spec: np.ndarray, n_bins: int = 64) -> dict[str, float]:
    spec = np.asarray(spec, dtype=np.float32)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

    x_old = np.linspace(0, 1, spec.size)
    x_new = np.linspace(0, 1, n_bins)
    resampled = np.interp(x_new, x_old, spec).astype(np.float32)

    out: dict[str, float] = {}
    for i, v in enumerate(resampled):
        out[f"f{i:03d}"] = float(v)

    return out
