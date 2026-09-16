import random

import pandas as pd
import torch
from two_tower import TwoTowerModel

import json


def main():
    with open("data/metadata.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    test_df = pd.read_csv("data/test.csv")

    embedding_dim = meta["embedding_dim"]
    model = TwoTowerModel(n_users, n_items, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load("model/two_tower.pt", weights_only=True))
    model.eval()

    K = 5
    RELEVANCE_THRESHOLD = 4.0

    random.seed(42)

    random_precisions = []
    with torch.no_grad():
        for user_idx, group in test_df.groupby('user_idx'):
            if len(group) <= K:
                continue
            actual_ratings = group['rating'].values
            random_idx = random.sample(range(len(group)), K)
            random_actual = actual_ratings[random_idx]
            hits = (random_actual >= RELEVANCE_THRESHOLD).sum()
            random_precisions.append(hits / K)

    avg_random = sum(random_precisions) / len(random_precisions)
    print(f"Random baseline Precision@{K}: {avg_random:.4f}")

    precisions = []
    with torch.no_grad():
        for user_idx, group in test_df.groupby('user_idx'):
            if len(group) <= K:
                continue

            item_ids = torch.tensor(group['item_idx'].values, dtype=torch.long)
            user_ids = torch.full_like(item_ids, user_idx)

            preds = model(user_ids, item_ids)
            top_k_idx = torch.topk(preds, K).indices

            actual_ratings = group['rating'].values
            top_k_actual = actual_ratings[top_k_idx.numpy()]

            hits = (top_k_actual >= RELEVANCE_THRESHOLD).sum()
            precisions.append(hits / K)

    avg_precision = sum(precisions) / len(precisions)
    print(f"Users evaluated: {len(precisions)}")
    print(f"Precision@{K}: {avg_precision:.4f}")


if __name__ == "__main__":
    main()