import pandas as pd

from codexspectra.evaluation.splits import make_splits


def test_make_splits_returns_all_labels():
    df = pd.DataFrame({"label": ["a", "b"] * 20, "group_id": [f"g{i//2}" for i in range(40)]})
    split = make_splits(df, grouped=True)
    assert set(split.unique()) == {"train", "val", "test"}
