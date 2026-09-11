import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


class FeedForward(nn.Module):
    def __init__(self, dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)
