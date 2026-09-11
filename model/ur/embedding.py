import math
import torch
import torch.nn as nn


class GlobalCoordinateEmbedding(nn.Module):
    """Normalized global coordinates embedding, using Fourier features."""
    def __init__(self, dim: int = 256, num_freqs: int = 64):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.linspace(0, 6, num_freqs))
        self.proj = nn.Linear(num_freqs * 4, dim)

    def forward(self, coord):
        x = coord[:, 0:1] * self.frequencies[None, :] * (2 * math.pi)
        y = coord[:, 1:2] * self.frequencies[None, :] * (2 * math.pi)
        features = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)
        return self.proj(features).unsqueeze(1)


class UREmbedding(nn.Module):
    """Convert UR1/UR2 to tokens, and add local positions, summary tokens, and global coordinates."""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.ur1_pos = nn.Parameter(torch.zeros(1, 32 * 32, dim))
        self.ur2_pos = nn.Parameter(torch.zeros(1, 16 * 16, dim))
        self.ur1_summary = nn.Parameter(torch.zeros(1, 1, dim))
        self.ur2_summary = nn.Parameter(torch.zeros(1, 1, dim))
        self.global_coord = GlobalCoordinateEmbedding(dim)

        nn.init.trunc_normal_(self.ur1_pos, std=0.02)
        nn.init.trunc_normal_(self.ur2_pos, std=0.02)
        nn.init.trunc_normal_(self.ur1_summary, std=0.02)
        nn.init.trunc_normal_(self.ur2_summary, std=0.02)

    def forward(self, ur1, ur2, coord):
        b = ur1.size(0)

        # [B,256,H,W] -> [B,HW,256]，add local position embedding
        x1 = ur1.flatten(2).transpose(1, 2) + self.ur1_pos
        x2 = ur2.flatten(2).transpose(1, 2) + self.ur2_pos

        # prepend summary token
        x1 = torch.cat([self.ur1_summary.expand(b, -1, -1), x1], dim=1)
        x2 = torch.cat([self.ur2_summary.expand(b, -1, -1), x2], dim=1)

        # add global position embedding
        global_pos = self.global_coord(coord)
        return x1 + global_pos, x2 + global_pos