from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize

from codexspectra.utils.config import load_config


def numeric_feature_cols(df):
    blocked = {
        "cube_id", "group_id", "label", "pixel_id", "split",
        "path", "checksum", "source_file", "material", "sample_id"
    }
    return [
        c for c in df.columns
        if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_path = Path(cfg["data"]["feature_table_path"])

    figures_dir = Path("reports/figures")
    tables_dir = Path("reports/tables")
    model_dir = Path("models")

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_path)
    feature_cols = numeric_feature_cols(df)

    model_path = model_dir / "best_tuned_random_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    model = joblib.load(model_path)

    # Class balance
    counts = df["label"].value_counts().sort_index()
    counts.to_csv(tables_dir / "class_balance.csv", header=["n"])

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class balance across sampled spectra")
    plt.xlabel("Class")
    plt.ylabel("Number of sampled spectra")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_balance.png", dpi=220)
    plt.close()

    # Split balance
    split_counts = pd.crosstab(df["split"], df["label"])
    split_counts.to_csv(tables_dir / "split_class_balance.csv")

    split_counts.plot(kind="bar", figsize=(9, 6))
    plt.title("Class balance by split")
    plt.xlabel("Split")
    plt.ylabel("Number of sampled spectra")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "split_class_balance.png", dpi=220)
    plt.close()

    # Mean feature profile
    means = df.groupby("label")[feature_cols].mean().T
    means.to_csv(tables_dir / "mean_feature_profile_by_class.csv")

    means.plot(figsize=(11, 6))
    plt.title("Mean spectral feature profile by class")
    plt.xlabel("Feature index")
    plt.ylabel("Mean feature value")
    plt.tight_layout()
    plt.savefig(figures_dir / "spectral_means_by_class.png", dpi=220)
    plt.close()

    # PCA
    sample = df.sample(min(len(df), 20000), random_state=42)
    X = sample[feature_cols]
    y = sample["label"].astype(str)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)

    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }).to_csv(tables_dir / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(8, 6))
    for lab in sorted(y.unique()):
        idx = y == lab
        plt.scatter(z[idx, 0], z[idx, 1], s=6, alpha=0.45, label=lab)
    plt.title("PCA projection of sampled spectra")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_latent_space.png", dpi=220)
    plt.close()

    # Overfitting diagnostic
    perf_rows = []
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        pred = model.predict(sdf[feature_cols])
        perf_rows.append({
            "split": split,
            "n": len(sdf),
            "macro_f1": f1_score(sdf["label"], pred, average="macro"),
            "weighted_f1": f1_score(sdf["label"], pred, average="weighted"),
        })

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(tables_dir / "overfitting_diagnostic.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(perf["split"], perf["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(perf["split"], perf["weighted_f1"], marker="o", label="Weighted-F1")
    plt.ylim(0, 1.05)
    plt.title("Overfitting diagnostic by split")
    plt.xlabel("Split")
    plt.ylabel("F1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "overfitting_diagnostic.png", dpi=220)
    plt.close()

    # Test metrics
    test = df[df["split"] == "test"]
    y_true = test["label"]
    y_pred = model.predict(test[feature_cols])
    labels = sorted(y_true.unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = normalize(cm, norm="l1", axis=1)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        tables_dir / "confusion_matrix_test_counts.csv"
    )
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        tables_dir / "confusion_matrix_test_normalized.csv"
    )

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, aspect="auto")
    plt.title("Normalized test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="Row-normalized proportion")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_test_normalized.png", dpi=220)
    plt.close()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(tables_dir / "classification_report_test_expanded.csv")

    class_rows = report_df.loc[
        [x for x in labels if x in report_df.index],
        ["precision", "recall", "f1-score"]
    ]

    class_rows.plot(kind="bar", figsize=(9, 6))
    plt.ylim(0, 1.05)
    plt.title("Per-class test precision, recall, and F1")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_test_metrics.png", dpi=220)
    plt.close()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(tables_dir / "feature_importance.csv", index=False)

        top = imp.head(25).iloc[::-1]
        plt.figure(figsize=(8, 8))
        plt.barh(top["feature"], top["importance"])
        plt.title("Top Random Forest feature importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=220)
        plt.close()

    # Per-group performance
    group_rows = []
    for group_id, gdf in test.groupby("group_id"):
        pred = model.predict(gdf[feature_cols])
        group_rows.append({
            "group_id": group_id,
            "n": len(gdf),
            "true_label": gdf["label"].mode().iloc[0],
            "macro_f1": f1_score(gdf["label"], pred, average="macro"),
            "accuracy": (gdf["label"].to_numpy() == pred).mean(),
        })

    group_perf = pd.DataFrame(group_rows).sort_values("macro_f1")
    group_perf.to_csv(tables_dir / "per_group_test_performance.csv", index=False)

    plot_group = group_perf.head(40).iloc[::-1]
    plt.figure(figsize=(9, 10))
    plt.barh(plot_group["group_id"].astype(str), plot_group["macro_f1"])
    plt.xlim(0, 1.05)
    plt.title("Lowest-performing test document groups")
    plt.xlabel("Macro-F1")
    plt.ylabel("Group ID")
    plt.tight_layout()
    plt.savefig(figures_dir / "lowest_performing_groups.png", dpi=220)
    plt.close()

    print("Generated figures:")
    for p in sorted(figures_dir.glob("*.png")):
        print("-", p)


if __name__ == "__main__":
    main()
