import pytest

import pandas as pd

try:
    import main
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    _train = pd.read_csv("data/train.csv")
    trained_item_idx = int(_train["item_idx"].iloc[0])
    del _train
except (FileNotFoundError, OSError) as e:
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


def test_every_in_range_user_has_training_rows():
    # prepare.py encodes user_idx/item_idx AFTER the >=5-ratings filter, so
    # indices are compact -- this was never an encode-order bug. The gap used
    # to come from the random split: a user with exactly the minimum 5 ratings
    # could have all of them land in test, leaving a valid in-range user_idx
    # with zero rows in train.csv and an untouched embedding behind it.
    #
    # The per-user temporal split holds out only a fraction of each user's
    # interactions, so every user keeps training rows and that gap can no
    # longer arise. This asserts the guarantee directly. The endpoint guards
    # stay as defense against metadata/checkpoint drift and are covered by the
    # out-of-range cases in `cases` above.
    _train = pd.read_csv("data/train.csv")
    trained_users = set(_train["user_idx"].unique().tolist())
    untrained = [u for u in range(int(main.n_users)) if u not in trained_users]

    assert not untrained, (
        f"{len(untrained)} in-range user_idx have no training rows "
        f"(first few: {untrained[:5]}); the split should leave every user "
        f"with train data, and the endpoints must reject any that slip through"
    )