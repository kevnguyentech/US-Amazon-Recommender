import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, global_bias_init=4.26):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        # defaults to the Amazon Electronics train-split mean; pass the
        # actual mean when training on other data
        self.global_bias = nn.Parameter(torch.tensor(global_bias_init))

        # small init so dot products don't start huge
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        dot = (user_vec * item_vec).sum(dim=1)
        bias = self.user_bias(user_idx).squeeze(1) + self.item_bias(item_idx).squeeze(1) + self.global_bias
        return dot + bias