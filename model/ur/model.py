import torch
import torch.nn as nn
from .embedding import UREmbedding
from .attention import SelfAttention, CrossAttention, FeedForward


class URModel(nn.Module):
    """Unified Representation Model for HE-UR1 and HE-UR2."""
    def __init__(self, dim=256, heads=8, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = UREmbedding(dim)

        self.self_attn1 = SelfAttention(dim, heads, dropout)
        self.self_attn2 = SelfAttention(dim, heads, dropout)
        self.self_norm1 = nn.LayerNorm(dim)
        self.self_norm2 = nn.LayerNorm(dim)
        self.self_ffn1 = FeedForward(dim, ffn_dim, dropout)
        self.self_ffn2 = FeedForward(dim, ffn_dim, dropout)
        self.self_ffn_norm1 = nn.LayerNorm(dim)
        self.self_ffn_norm2 = nn.LayerNorm(dim)

        self.cross12 = CrossAttention(dim, heads, dropout)
        self.cross21 = CrossAttention(dim, heads, dropout)
        self.cross_norm1 = nn.LayerNorm(dim)
        self.cross_norm2 = nn.LayerNorm(dim)
        self.cross_ffn1 = FeedForward(dim, ffn_dim, dropout)
        self.cross_ffn2 = FeedForward(dim, ffn_dim, dropout)
        self.cross_ffn_norm1 = nn.LayerNorm(dim)
        self.cross_ffn_norm2 = nn.LayerNorm(dim)

        self.concept_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
            nn.Sigmoid()
        )

    def encode(self, ur1, ur2, coord, return_tokens=False):
        """
        UR1 [B,256,32,32] + UR2 [B,256,16,16]
        -> fused UR [B,512]
        """
        if ur1.ndim != 4 or ur1.shape[1:] != (256, 32, 32):
            raise ValueError(f"Expected UR1 [B,256,32,32], got {tuple(ur1.shape)}")
        if ur2.ndim != 4 or ur2.shape[1:] != (256, 16, 16):
            raise ValueError(f"Expected UR2 [B,256,16,16], got {tuple(ur2.shape)}")

        x1, x2 = self.embedding(ur1, ur2, coord)

        # self-attention
        x1 = x1 + self.self_attn1(self.self_norm1(x1))
        x2 = x2 + self.self_attn2(self.self_norm2(x2))
        x1 = x1 + self.self_ffn1(self.self_ffn_norm1(x1))
        x2 = x2 + self.self_ffn2(self.self_ffn_norm2(x2))

        # cross-attention, based on self-attended representation
        n1, n2 = self.cross_norm1(x1), self.cross_norm2(x2)
        x1 = x1 + self.cross12(n1, n2)
        x2 = x2 + self.cross21(n2, n1)

        x1 = x1 + self.cross_ffn1(self.cross_ffn_norm1(x1))
        x2 = x2 + self.cross_ffn2(self.cross_ffn_norm2(x2))

        # summary token → unified representation
        fused = torch.cat([x1[:, 0], x2[:, 0]], dim=-1)  # [B,512]

        if return_tokens:
            return fused, x1[:, 1:], x2[:, 1:]
        return fused

    def forward(self, ur1, ur2, coord):
        fused = self.encode(ur1, ur2, coord)
        return self.concept_head(fused)