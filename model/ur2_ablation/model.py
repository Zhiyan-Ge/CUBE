import torch.nn as nn
from .embedding import UR2Embedding
from .attention import SelfAttention, FeedForward


class UR2AblationModel(nn.Module):
    """Architecture-matched UR2-only baseline: self-attn -> FFN -> self-attn -> FFN."""
    def __init__(self, dim=256, heads=8, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = UR2Embedding(dim)
        self.attn1, self.attn2 = SelfAttention(dim,heads,dropout), SelfAttention(dim,heads,dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn1, self.ffn2 = FeedForward(dim,ffn_dim,dropout), FeedForward(dim,ffn_dim,dropout)
        self.ffn_norm1, self.ffn_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.concept_head = nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,3),nn.Sigmoid())

    def encode(self, ur, coord, return_tokens=False):
        if ur.ndim!=4 or ur.shape[1:]!=(256,16,16): raise ValueError(f"Expected UR2 [B,256,16,16], got {tuple(ur.shape)}")
        x = self.embedding(ur,coord)
        x = x + self.attn1(self.norm1(x)); x = x + self.ffn1(self.ffn_norm1(x))
        x = x + self.attn2(self.norm2(x)); x = x + self.ffn2(self.ffn_norm2(x))
        return (x[:,0],x[:,1:]) if return_tokens else x[:,0]

    def forward(self, ur, coord): return self.concept_head(self.encode(ur,coord))
