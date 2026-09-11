import torch.nn as nn


class SelfAttention(nn.Module):
    """Single-sequence self-attention: query/key/value all come from the same UR."""
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


class CrossAttention(nn.Module):
    """Cross-sequence attention: query comes from one UR, key/value come from another UR."""
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, query, context):
        return self.attn(query, context, context, need_weights=False)[0]


class FeedForward(nn.Module):
    """Feed-forward network: applied after attention."""
    def __init__(self, dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)