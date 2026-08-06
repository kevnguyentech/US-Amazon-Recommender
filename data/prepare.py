# data/prepare.py
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/amazon_reviews.csv")

    while True:
        prev_len = len(df)
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= 5].index)]
        if len(df) == prev_len:
            break

    # encode AFTER filtering so indices are compact (0..54k, not sparse up to ~150k)
    df['user_idx'] = df['user_id'].astype('category').cat.codes
    df['item_idx'] = df['item_id'].astype('category').cat.codes

    n_users = int(df['user_idx'].nunique())
    n_items = int(df['item_idx'].nunique())
    print(f"Users after filtering: {n_users}, Items: {n_items}")

    existing_meta = {}
    if os.path.exists("data/metadata.json"):
        with open("data/metadata.json") as f:
            existing_meta = json.load(f)

    with open("data/metadata.json", "w") as f:
        json.dump({**existing_meta, "n_users": n_users, "n_items": n_items}, f)
    print("Saved metadata.json")

    print(f"After filtering: {df.shape}")
    print(f"Items after filtering: {df['item_id'].nunique()}")
    print(f"Avg ratings per item: {len(df) / df['item_id'].nunique():.2f}")

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"Train: {len(train)}, Test: {len(test)}")


if __name__ == "__main__":
    main()