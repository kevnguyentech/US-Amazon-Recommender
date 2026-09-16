import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import RatingsDataset
from two_tower import TwoTowerModel

import json


def main():
    with open("data/metadata.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_ds = RatingsDataset(train_df)
    test_ds = RatingsDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    EMBEDDING_DIM = 16
    model = TwoTowerModel(n_users, n_items, embedding_dim=EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    EPOCHS = 3

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_train_samples = 0
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            preds = model(users, items)
            loss = loss_fn(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)
            total_train_samples += len(ratings)

        avg_loss = total_loss / total_train_samples
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    model.eval()
    total_test_loss = 0
    total_samples = 0
    with torch.no_grad():
        for users, items, ratings in test_loader:
            preds = model(users, items)
            loss = loss_fn(preds, ratings)
            total_test_loss += loss.item() * len(ratings)
            total_samples += len(ratings)

    avg_test_loss = total_test_loss / total_samples
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"Test RMSE: {avg_test_loss ** 0.5:.4f}")

    torch.save(model.state_dict(), "model/two_tower.pt")
    print("Model saved.")

    with open("data/metadata.json") as f:
        saved_meta = json.load(f)
    saved_meta["embedding_dim"] = EMBEDDING_DIM
    with open("data/metadata.json", "w") as f:
        json.dump(saved_meta, f)


if __name__ == "__main__":
    main()