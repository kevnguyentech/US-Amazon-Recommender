import pandas as pd
import pytest

try:
    import main
    from fastapi.testclient import TestClient
except (FileNotFoundError, OSError) as e:
    pytest.skip(
        f"API could not be imported -- model likely not trained yet: {e}",
        allow_module_level=True,
    )


def test_recommendations_are_trained_items():
    train_df = pd.read_csv("data/train.csv")
    valid_ids = set(train_df["item_idx"].unique())

    client = TestClient(main.app)

    checked = 0
    for user_idx in range(30):
        r = client.post("/recommend", json={"user_idx": user_idx, "top_k": 5})
        if r.status_code != 200:
            continue
        checked += 1
        recs = [x["item_idx"] for x in r.json()["recommendations"]]
        bad = [i for i in recs if i not in valid_ids]
        assert not bad, f"user {user_idx} got untrained items: {bad}"
    assert checked > 0, "no user returned 200; test proved nothing"


def test_recommendations_exclude_already_rated():
    train_df = pd.read_csv("data/train.csv")
    seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    client = TestClient(main.app)

    checked = 0
    for user_idx in range(200):
        r = client.post("/recommend", json={"user_idx": user_idx, "top_k": 5})
        if r.status_code != 200:
            continue
        checked += 1
        recs = {x["item_idx"] for x in r.json()["recommendations"]}
        overlap = recs & seen.get(user_idx, set())
        assert not overlap, f"user {user_idx} was recommended already-rated items: {overlap}"
    assert checked > 0, "no user returned 200; test proved nothing"
