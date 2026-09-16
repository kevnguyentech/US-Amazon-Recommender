import pandas as pd
import pytest


# ── Synthetic data ─────────────────────────────────────────────────────────────
# Users A-E have 5 ratings each (survive the >=5 filter).
# Users F-H have 2 ratings each (get filtered out).
# Items X-Z have enough ratings from A-E alone (5 each) to survive.

def _synthetic_df():
    rows = []
    for user in ["A", "B", "C", "D", "E"]:
        for item in ["X", "Y", "Z", "W", "V"]:
            rows.append({"user_id": user, "item_id": item, "rating": 4.0, "timestamp": 0})
    for user in ["F", "G", "H"]:
        rows.append({"user_id": user, "item_id": "X", "rating": 3.0, "timestamp": 0})
        rows.append({"user_id": user, "item_id": "Y", "rating": 3.0, "timestamp": 0})
    return pd.DataFrame(rows)


def _apply_prepare_logic(df):
    """Mirror of the core logic in data/prepare.py."""
    while True:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= 5].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= 5].index)]
        if len(df) == prev_len:
            break
    df = df.copy()
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["item_id"].astype("category").cat.codes
    return df


# ── Filtering correctness ──────────────────────────────────────────────────────

def test_sparse_users_removed():
    filtered = _apply_prepare_logic(_synthetic_df())
    assert set(filtered["user_id"].unique()) == {"A", "B", "C", "D", "E"}
    for dropped in ["F", "G", "H"]:
        assert dropped not in filtered["user_id"].values


def test_all_valid_user_ratings_retained():
    filtered = _apply_prepare_logic(_synthetic_df())
    for user in ["A", "B", "C", "D", "E"]:
        assert len(filtered[filtered["user_id"] == user]) == 5


# ── Index compactness (the core regression for Finding 1) ─────────────────────

def test_user_indices_compact():
    """user_idx must be exactly 0..n_users-1 with no gaps."""
    filtered = _apply_prepare_logic(_synthetic_df())
    n_users = filtered["user_idx"].nunique()
    assert set(filtered["user_idx"].unique()) == set(range(n_users))


def test_item_indices_compact():
    """item_idx must be exactly 0..n_items-1 with no gaps."""
    filtered = _apply_prepare_logic(_synthetic_df())
    n_items = filtered["item_idx"].nunique()
    assert set(filtered["item_idx"].unique()) == set(range(n_items))


def test_user_idx_max_equals_nunique_minus_one():
    filtered = _apply_prepare_logic(_synthetic_df())
    assert filtered["user_idx"].max() == filtered["user_idx"].nunique() - 1


def test_item_idx_max_equals_nunique_minus_one():
    filtered = _apply_prepare_logic(_synthetic_df())
    assert filtered["item_idx"].max() == filtered["item_idx"].nunique() - 1


def test_stale_index_regression():
    """
    Regression: if indices are assigned BEFORE filtering, max(user_idx) > nunique-1
    because filtered-out users leave gaps. Encoding after filtering prevents this.
    
    Uses interleaved user names so dropped users (B, D, F, H, J) have codes
    1, 3, 5, 7, 9 and kept users (A, C, E, G, I) have codes 0, 2, 4, 6, 8 --
    demonstrating the gap.

    Five kept users are needed, not three: with only three, every item holds
    just three ratings once the sparse users are filtered out, so a correct
    5-core pass empties the frame entirely and there are no indices to compare.
    """
    rows = []
    for user in ["A", "C", "E", "G", "I"]:  # kept: >=5 ratings each
        for item in ["X", "Y", "Z", "W", "V"]:
            rows.append({"user_id": user, "item_id": item, "rating": 4.0})
    for user in ["B", "D", "F", "H", "J"]:  # dropped: only 2 ratings each
        rows.append({"user_id": user, "item_id": "X", "rating": 3.0})
        rows.append({"user_id": user, "item_id": "Y", "rating": 3.0})
    df = pd.DataFrame(rows)

    # Simulate OLD (broken) behavior: encode first, filter second
    df["user_idx_stale"] = df["user_id"].astype("category").cat.codes
    user_counts = df["user_id"].value_counts()
    df_filtered = df[df["user_id"].isin(user_counts[user_counts >= 5].index)].copy()

    stale_max = df_filtered["user_idx_stale"].max()
    stale_nunique = df_filtered["user_idx_stale"].nunique()

    # A=0, B=1, ... J=9 -- dropping B,D,F,H,J leaves codes 0,2,4,6,8: max=8 > nunique-1=4
    assert stale_max > stale_nunique - 1, (
        "Stale approach should produce gaps when dropped users are interleaved alphabetically"
    )

    # Correct approach (encode after filtering): compact 0..n-1, no gaps
    correct = _apply_prepare_logic(df)
    assert correct["user_idx"].max() == correct["user_idx"].nunique() - 1


def test_five_core_converges():
    """One filtering pass is not enough: dropping sparse users can starve
    an item below the threshold, which must then drop too."""
    rows = []
    for user in ["A", "B", "C", "D", "E"]:
        for item in ["X", "Y", "Z", "W", "V"]:
            rows.append({"user_id": user, "item_id": item, "rating": 4.0})
    for user in ["F", "G", "H", "I", "J"]:      # 1 rating each, all dropped
        rows.append({"user_id": user, "item_id": "Q", "rating": 3.0})
    rows.append({"user_id": "A", "item_id": "Q", "rating": 5.0})

    filtered = _apply_prepare_logic(pd.DataFrame(rows))

    assert "Q" not in filtered["item_id"].values
    assert (filtered["item_id"].value_counts() >= 5).all()
    assert (filtered["user_id"].value_counts() >= 5).all()
