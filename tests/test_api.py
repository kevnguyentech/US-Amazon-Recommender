import pandas as pd
import pytest

# Skip the entire module if the model hasn't been trained yet
# (missing two_tower.pt or data files). Run `python model/train.py` first.
try:
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
except (FileNotFoundError, OSError) as e:
    pytest.skip(
        f"API could not be imported -- model likely not trained yet: {e}",
        allow_module_level=True,
    )

# Pull one valid user/item from train.csv once at module level
# so individual tests don't each re-read the CSV.
_train = pd.read_csv("data/train.csv")
VALID_USER = int(_train["user_idx"].iloc[0])
VALID_ITEM = int(_train["item_idx"].iloc[0])
INVALID_IDX = 999_999


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_valid_returns_200():
    resp = client.post("/predict", json={"user_idx": VALID_USER, "item_idx": VALID_ITEM})
    assert resp.status_code == 200


def test_predict_response_has_predicted_rating():
    resp = client.post("/predict", json={"user_idx": VALID_USER, "item_idx": VALID_ITEM})
    assert "predicted_rating" in resp.json()
    assert isinstance(resp.json()["predicted_rating"], float)


def test_predict_invalid_user_returns_400():
    resp = client.post("/predict", json={"user_idx": INVALID_IDX, "item_idx": VALID_ITEM})
    assert resp.status_code == 400


def test_predict_invalid_item_returns_400():
    resp = client.post("/predict", json={"user_idx": VALID_USER, "item_idx": INVALID_IDX})
    assert resp.status_code == 400


# ── /recommend ────────────────────────────────────────────────────────────────

def test_recommend_valid_returns_200():
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": 5})
    assert resp.status_code == 200


def test_recommend_returns_correct_count():
    top_k = 5
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": top_k})
    recs = resp.json()["recommendations"]
    assert len(recs) == top_k


def test_recommend_items_have_expected_keys():
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": 3})
    for rec in resp.json()["recommendations"]:
        assert "item_idx" in rec
        assert "predicted_rating" in rec


def test_recommend_invalid_user_returns_400():
    resp = client.post("/recommend", json={"user_idx": INVALID_IDX, "top_k": 5})
    assert resp.status_code == 400


def test_recommend_top_k_zero_returns_400():
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": 0})
    assert resp.status_code == 400


def test_recommend_top_k_too_large_returns_400():
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": INVALID_IDX})
    assert resp.status_code == 400


def test_recommend_scores_descending():
    resp = client.post("/recommend", json={"user_idx": VALID_USER, "top_k": 10})
    scores = [r["predicted_rating"] for r in resp.json()["recommendations"]]
    assert scores == sorted(scores, reverse=True)