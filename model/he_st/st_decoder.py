import torch.nn as nn
from .blocks import ResidualBlock


class STDecoder(nn.Module):
    """Decode UR2 to pseudo-ST."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualBlock(256, groups), ResidualBlock(256, groups)
        )
        self.channel_proj = nn.Conv2d(256, 256, 1)
        self.output = nn.Linear(256, 256)

    def forward(self, ur2):
        x = self.channel_proj(self.decoder(ur2))
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.output(x)