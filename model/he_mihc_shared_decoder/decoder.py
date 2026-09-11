import torch.nn as nn
from .blocks import UpsampleBlock


class ImageDecoder(nn.Module):
    """Decode UR1 features to image."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            UpsampleBlock(256, 128, groups),
            UpsampleBlock(128, 64, groups),
            UpsampleBlock(64, 32, groups),
            UpsampleBlock(32, 32, groups),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, ur1):
        return self.decoder(ur1)