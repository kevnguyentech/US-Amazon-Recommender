import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from two_tower import TwoTowerModel

app = FastAPI()

import json

with open("data/metadata.json") as f:
    meta = json.load(f)
n_users = meta["n_users"]
n_items = meta["n_items"]

train_df = pd.read_csv("data/train.csv")
valid_item_idx = torch.tensor(sorted(train_df['item_idx'].unique()), dtype=torch.long)
# Validate against the actual set of trained indices rather than
# the raw [0, n_users) range. Protects against metadata/checkpoint
# drift if the model is ever retrained on a subset of users or items.
valid_user_idx_set = set(train_df['user_idx'].unique().tolist())
valid_item_idx_set = set(valid_item_idx.tolist())
# item positions (not ids) each user already rated, for exclusion at serve time
_pos_of_item = {item: pos for pos, item in enumerate(valid_item_idx.tolist())}
seen_positions = (
    train_df.groupby('user_idx')['item_idx']
    .apply(lambda s: [_pos_of_item[i] for i in s.unique()])
    .to_dict()
)
del train_df

embedding_dim = meta["embedding_dim"]
model = TwoTowerModel(n_users, n_items, embedding_dim=embedding_dim)
model.load_state_dict(torch.load("model/two_tower.pt", weights_only=True))
model.eval()

class PredictRequest(BaseModel):
    user_idx: int
    item_idx: int

@app.post("/predict")
def predict(req: PredictRequest):
    if req.user_idx not in valid_user_idx_set:
        raise HTTPException(status_code=400, detail=f"user_idx {req.user_idx} was never trained (not in training data)")
    if req.item_idx not in valid_item_idx_set:
        raise HTTPException(status_code=400, detail=f"item_idx {req.item_idx} was never trained (not in training data)")
    with torch.no_grad():
        user_tensor = torch.tensor([req.user_idx], dtype=torch.long)
        item_tensor = torch.tensor([req.item_idx], dtype=torch.long)
        pred = model(user_tensor, item_tensor)
    return {"predicted_rating": float(pred.item())}

class RecommendRequest(BaseModel):
    user_idx: int
    top_k: int = 10

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_idx not in valid_user_idx_set:
        raise HTTPException(status_code=400, detail=f"user_idx {req.user_idx} was never trained (not in training data)")
    if not (1 <= req.top_k <= len(valid_item_idx)):
        raise HTTPException(status_code=400, detail=f"top_k must be in [1, {len(valid_item_idx)}]")
    all_items = valid_item_idx
    user_tensor = torch.full_like(all_items, req.user_idx)

    with torch.no_grad():
        preds = model(user_tensor, all_items)

    seen = seen_positions.get(req.user_idx, [])
    if len(valid_item_idx) - len(seen) < req.top_k:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be in [1, {len(valid_item_idx) - len(seen)}] for this user "
                   f"({len(seen)} items already rated are excluded)",
        )
    preds[seen] = float('-inf')

    top_k = torch.topk(preds, req.top_k)
    top_items = valid_item_idx[top_k.indices].tolist()
    top_scores = top_k.values.tolist()

    return {
        "user_idx": req.user_idx,
        "recommendations": [
            {"item_idx": item, "predicted_rating": score}
            for item, score in zip(top_items, top_scores)
        ]
    }