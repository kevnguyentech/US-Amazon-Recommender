import pytest

import pandas as pd

try:
    import main
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    _train = pd.read_csv("data/train.csv")
    trained_item_idx = int(_train["item_idx"].iloc[0])
    del _train
except Exception as e:
    pytest.skip(
        f"API could not be imported -- model likely not trained yet: {e}",
        allow_module_level=True,
    )

cases = [
    ("predict", {"user_idx": -1, "item_idx": 0}, 400),
    ("predict", {"user_idx": int(main.n_users) + 100, "item_idx": 0}, 400),
    ("predict", {"user_idx": 0, "item_idx": -1}, 400),
    ("predict", {"user_idx": 0, "item_idx": trained_item_idx}, 200),
    ("recommend", {"user_idx": -1, "top_k": 5}, 400),
    ("recommend", {"user_idx": 0, "top_k": 0}, 400),
    ("recommend", {"user_idx": 0, "top_k": 5}, 200),
]


@pytest.mark.parametrize("endpoint,payload,expected", cases)
def test_endpoint_validation(endpoint, payload, expected):
    r = client.post(f"/{endpoint}", json=payload)
    assert r.status_code == expected


def test_untrained_but_in_range_index_is_rejected():
    # prepare.py already encodes user_idx/item_idx AFTER the >=5-ratings
    # filter, so indices are compact -- this is not an encode-order bug.
    # The gap comes from train_test_split: a user with exactly the minimum
    # 5 ratings can have all of them land in the test split, leaving a
    # valid in-range user_idx with zero rows in train.csv. Both endpoints
    # must reject that index instead of silently predicting from an
    # untouched embedding.
    _train = pd.read_csv("data/train.csv")
    trained_users = set(_train["user_idx"].unique().tolist())
    untrained = next((u for u in range(int(main.n_users)) if u not in trained_users), None)
    if untrained is None:
        pytest.skip("no untrained-but-in-range user_idx in this dataset")

    r_predict = client.post("/predict", json={"user_idx": untrained, "item_idx": 0})
    assert r_predict.status_code == 400

    r_recommend = client.post("/recommend", json={"user_idx": untrained, "top_k": 5})
    assert r_recommend.status_code == 400