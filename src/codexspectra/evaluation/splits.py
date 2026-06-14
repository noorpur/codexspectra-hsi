from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "group_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 2026,
    grouped: bool = True,
) -> pd.Series:
    y = df[label_col].astype(str)
    idx = np.arange(len(df))
    split = pd.Series("train", index=df.index, name="split")
    if grouped and group_col in df.columns and df[group_col].nunique() > 3:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss.split(idx, y, groups=df[group_col]))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
        train_idx_rel, val_idx_rel = next(
            gss2.split(train_val_idx, y.iloc[train_val_idx], groups=df[group_col].iloc[train_val_idx])
        )
        val_idx = train_val_idx[val_idx_rel]
        train_idx = train_val_idx[train_idx_rel]
    else:
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=seed + 1,
            stratify=y.iloc[train_val_idx],
        )
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split
