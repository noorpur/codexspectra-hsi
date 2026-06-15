#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


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

    # HYPERDOC examples: DataCube shape = (bands, width, height)
    if cube.shape[1] == w and cube.shape[2] == h:
        return np.transpose(cube, (2, 1, 0))

    # Some HSI datasets use (bands, height, width)
    if cube.shape[1] == h and cube.shape[2] == w:
        return np.transpose(cube, (1, 2, 0))

    # Already HWC
    if cube.shape[0] == h and cube.shape[1] == w:
        return cube

    raise ValueError(f"Cannot align cube shape {cube.shape} with mask shape {mask_shape}")


def spectrum_features(spec: np.ndarray, n_bins: int = 64) -> dict[str, float]:
    spec = np.asarray(spec, dtype=np.float32)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

    if spec.size < 2:
        raise ValueError("Spectrum too short")

    x_old = np.linspace(0, 1, spec.size)
    x_new = np.linspace(0, 1, n_bins)
    resampled = np.interp(x_new, x_old, spec).astype(np.float32)

    d1 = np.diff(resampled)

    out: dict[str, float] = {}
    for i, v in enumerate(resampled):
        out[f"f{i:03d}"] = float(v)

    out["stat_mean"] = float(np.mean(spec))
    out["stat_std"] = float(np.std(spec))
    out["stat_min"] = float(np.min(spec))
    out["stat_max"] = float(np.max(spec))
    out["stat_q10"] = float(np.quantile(spec, 0.10))
    out["stat_q50"] = float(np.quantile(spec, 0.50))
    out["stat_q90"] = float(np.quantile(spec, 0.90))
    out["stat_d1_mean"] = float(np.mean(d1))
    out["stat_d1_std"] = float(np.std(d1))

    return out


def grouped_split(groups: pd.Series, seed: int) -> dict[str, str]:
    unique_df = pd.DataFrame({"group_id": sorted(groups.astype(str).unique())})

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(unique_df, groups=unique_df["group_id"]))

    train_val = unique_df.iloc[train_val_idx].copy()
    test = unique_df.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    train_idx, val_idx = next(splitter2.split(train_val, groups=train_val["group_id"]))

    split_map = {}
    for g in train_val.iloc[train_idx]["group_id"]:
        split_map[str(g)] = "train"
    for g in train_val.iloc[val_idx]["group_id"]:
        split_map[str(g)] = "val"
    for g in test["group_id"]:
        split_map[str(g)] = "test"

    return split_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw/hyperdoc")
    parser.add_argument("--out", default="data/processed/hyperdoc_gt_material_m2/features.csv")
    parser.add_argument("--max-per-mask-label-per-cube", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--label-mode", choices=["category", "class_name"], default="category")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    raw_root = Path(args.raw_root)
    h5_dir = raw_root / "minicubes"
    gt_dir = raw_root / "GT" / "GT"
    label_path = raw_root / "Materials_label_and_colormap_assignation.ods"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = load_label_table(label_path)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"No .h5 files found in {h5_dir}")

    rows = []

    for h5_path in tqdm(h5_files, desc="Building GT-material features"):
        stem = h5_path.stem
        parts = stem.split("-")
        if len(parts) < 3:
            continue

        doc_id = parts[0]
        sensor = parts[1]
        doc_type = "-".join(parts[2:])

        gt_path = gt_dir / f"{doc_id}-{doc_type}_GT.png"
        if not gt_path.exists():
            continue

        mask = np.array(Image.open(gt_path))

        with h5py.File(h5_path, "r") as f:
            cube = f["DataCube"][()]

        hwc = cube_to_hwc(cube, mask.shape)

        for mask_value in sorted(np.unique(mask)):
            mask_value_int = int(mask_value)
            if mask_value_int == 0:
                continue
            if mask_value_int not in label_map:
                continue

            meta = label_map[mask_value_int]
            label = meta[args.label_mode].strip()
            if not label:
                continue

            ys, xs = np.where(mask == mask_value)
            n = len(ys)
            if n == 0:
                continue

            take = min(args.max_per_mask_label_per_cube, n)
            idx = rng.choice(n, size=take, replace=False)

            for local_i in idx:
                y = int(ys[local_i])
                x = int(xs[local_i])
                spec = hwc[y, x, :]

                feat = spectrum_features(spec)
                feat.update({
                    "cube_id": stem,
                    "group_id": doc_id,
                    "sensor": sensor,
                    "document_type": doc_type,
                    "pixel_y": y,
                    "pixel_x": x,
                    "mask_value": mask_value_int,
                    "label": label,
                    "material_category": meta["category"],
                    "material_class": meta["class_name"],
                    "material_subtype": meta["subtype"],
                })
                rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows were generated. Check GT paths and label mapping.")

    split_map = grouped_split(df["group_id"], args.seed)
    df["split"] = df["group_id"].astype(str).map(split_map)

    df.to_csv(out_path, index=False)

    label_table = pd.DataFrame.from_dict(label_map, orient="index")
    label_table.index.name = "mask_value"
    label_table.to_csv(out_path.parent / "mask_value_label_map.csv")

    print("Wrote:", out_path)
    print("Rows:", len(df))
    print()
    print("Labels:")
    print(df["label"].value_counts())
    print()
    print("Splits:")
    print(df["split"].value_counts())
    print()
    print("Groups per split:")
    print(df.groupby("split")["group_id"].nunique())


if __name__ == "__main__":
    main()
