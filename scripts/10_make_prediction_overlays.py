#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from scripts_compat import spectrum_features, cube_to_hwc, load_label_table


LABEL_COLORS = {
    "Ink": 1,
    "Pencil": 2,
    "Pigment": 3,
    "Substrate": 4,
}


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def colorize(label_map: np.ndarray, labels: list[str]) -> np.ndarray:
    palette = {
        "Ink": np.array([30, 90, 180], dtype=np.uint8),
        "Pencil": np.array([240, 160, 40], dtype=np.uint8),
        "Pigment": np.array([80, 170, 80], dtype=np.uint8),
        "Substrate": np.array([210, 70, 70], dtype=np.uint8),
        "Unknown": np.array([120, 120, 120], dtype=np.uint8),
    }

    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        out[label_map == i] = palette.get(label, palette["Unknown"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw/hyperdoc")
    parser.add_argument("--features", default="data/processed/hyperdoc_gt_material_m2/features.csv")
    parser.add_argument("--model", default="models/best_tuned_random_forest.joblib")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--sensor", default="VNIR")
    parser.add_argument("--out-dir", default="reports/figures/overlays")
    parser.add_argument("--max-pixels", type=int, default=90000)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    model = joblib.load(args.model)

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = [c for c in df.columns if c.startswith("f")]

    if args.doc_id is None:
        # Choose a test document with several labels if possible.
        test = df[df["split"] == "test"]
        doc_id = str(test.groupby("group_id")["label"].nunique().sort_values(ascending=False).index[0]).zfill(5)
    else:
        doc_id = str(args.doc_id).zfill(5)

    matches = sorted((raw_root / "minicubes").glob(f"{doc_id}-{args.sensor}-*.h5"))
    if not matches:
        matches = sorted((raw_root / "minicubes").glob(f"{doc_id}-*.h5"))
    if not matches:
        raise SystemExit(f"No H5 cube found for doc_id={doc_id}")

    h5_path = matches[0]
    stem = h5_path.stem
    parts = stem.split("-")
    doc_type = "-".join(parts[2:])
    sensor = parts[1]

    gt_path = raw_root / "GT" / "GT" / f"{doc_id}-{doc_type}_GT.png"
    rgb_path = raw_root / "RGB" / "RGB" / f"{doc_id}-{sensor}-{doc_type}.png"

    if not gt_path.exists():
        raise SystemExit(f"Missing GT mask: {gt_path}")

    gt = np.array(Image.open(gt_path))
    rgb = load_rgb(rgb_path)

    with h5py.File(h5_path, "r") as f:
        cube = f["DataCube"][()]
    hwc = cube_to_hwc(cube, gt.shape)

    h, w = gt.shape
    yy, xx = np.where(gt > 0)

    if len(yy) > args.max_pixels:
        rng = np.random.default_rng(1337)
        idx = rng.choice(len(yy), size=args.max_pixels, replace=False)
        yy = yy[idx]
        xx = xx[idx]

    rows = []
    coords = []
    for y, x in zip(yy, xx):
        spec = hwc[int(y), int(x), :]
        feat = spectrum_features(spec)
        rows.append(feat)
        coords.append((int(y), int(x)))

    X = pd.DataFrame(rows)
    pred = model.predict(X[feature_cols])
    labels = sorted(list(pd.Series(pred).unique()))

    # Keep stable label order
    all_labels = ["Ink", "Pencil", "Pigment", "Substrate"]
    labels = [x for x in all_labels if x in set(pred)] + [x for x in sorted(set(pred)) if x not in all_labels]

    pred_idx = np.zeros((h, w), dtype=np.int16) - 1
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for (y, x), label in zip(coords, pred):
        pred_idx[y, x] = label_to_idx[label]

    pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    visible = pred_idx >= 0
    pred_rgb[visible] = colorize(pred_idx.clip(min=0), labels)[visible]

    # GT category map
    label_table = load_label_table(raw_root / "Materials_label_and_colormap_assignation.ods")
    gt_label_idx = np.zeros((h, w), dtype=np.int16) - 1
    for value in np.unique(gt):
        value = int(value)
        if value in label_table:
            label = label_table[value]["category"]
            if label in label_to_idx:
                gt_label_idx[gt == value] = label_to_idx[label]

    gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt_visible = gt_label_idx >= 0
    gt_rgb[gt_visible] = colorize(gt_label_idx.clip(min=0), labels)[gt_visible]

    # Confidence map if available
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[feature_cols])
        conf_vals = proba.max(axis=1)
        conf = np.zeros((h, w), dtype=np.float32)
        for (y, x), c in zip(coords, conf_vals):
            conf[y, x] = c

    fig_path = out_dir / f"{doc_id}_{sensor}_{doc_type}_overlay_panel.png"

    if rgb is not None:
        panels = 4 if conf is not None else 3
        plt.figure(figsize=(5 * panels, 5))
        plt.subplot(1, panels, 1)
        plt.imshow(rgb)
        plt.title("RGB preview")
        plt.axis("off")

        plt.subplot(1, panels, 2)
        plt.imshow(gt_rgb)
        plt.title("GT material mask")
        plt.axis("off")

        plt.subplot(1, panels, 3)
        plt.imshow(pred_rgb)
        plt.title("Predicted material map")
        plt.axis("off")

        if conf is not None:
            plt.subplot(1, panels, 4)
            plt.imshow(conf, vmin=0, vmax=1)
            plt.title("Prediction confidence")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
    else:
        panels = 3 if conf is not None else 2
        plt.figure(figsize=(5 * panels, 5))

        plt.subplot(1, panels, 1)
        plt.imshow(gt_rgb)
        plt.title("GT material mask")
        plt.axis("off")

        plt.subplot(1, panels, 2)
        plt.imshow(pred_rgb)
        plt.title("Predicted material map")
        plt.axis("off")

        if conf is not None:
            plt.subplot(1, panels, 3)
            plt.imshow(conf, vmin=0, vmax=1)
            plt.title("Prediction confidence")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle(f"HYPERDOC material prediction overlay: {stem}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    print("Document:", doc_id)
    print("Cube:", h5_path)
    print("GT:", gt_path)
    print("RGB:", rgb_path if rgb_path.exists() else "missing")
    print("Labels:", labels)
    print("Wrote:", fig_path)


if __name__ == "__main__":
    main()
