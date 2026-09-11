import torch.nn as nn
from .blocks import ResidualBlock


class STEncoder(nn.Module):
    """Encode pseudo-ST to UR2."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.GroupNorm(groups, 256), nn.GELU()
        )
        self.encoder = nn.Sequential(
            ResidualBlock(256, groups), ResidualBlock(256, groups)
        )

    def forward(self, st):
        x = st.permute(0, 3, 1, 2).contiguous()
        return self.encoder(self.input_proj(x))