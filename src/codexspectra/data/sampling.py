from __future__ import annotations

import numpy as np
import pandas as pd

from codexspectra.data.loaders import load_cube


def cube_to_spectra(cube: np.ndarray, max_pixels: int, seed: int) -> np.ndarray:
    h, w, b = cube.shape
    flat = cube.reshape(h * w, b)
    valid = np.isfinite(flat).all(axis=1)
    flat = flat[valid]
    if len(flat) > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), size=max_pixels, replace=False)
        flat = flat[idx]
    return flat.astype(np.float32)


def build_feature_table(manifest: pd.DataFrame, max_pixels_per_cube: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, rec in manifest.reset_index(drop=True).iterrows():
        try:
            cube, _ = load_cube(rec.path)
            spectra = cube_to_spectra(cube, max_pixels=max_pixels_per_cube, seed=seed + i)
        except Exception as exc:
            print(f"[warn] skipped {rec.path}: {exc}")
            continue
        for j, spec in enumerate(spectra):
            row = {
                "cube_id": rec.cube_id,
                "group_id": rec.group_id,
                "label": rec.label,
                "pixel_id": j,
            }
            row.update({f"b{k:04d}": float(v) for k, v in enumerate(spec)})
            rows.append(row)
    return pd.DataFrame(rows)
