from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def baseline_models(seed: int = 1337, n_jobs: int = -1) -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "linear_svm": LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000),
    }
