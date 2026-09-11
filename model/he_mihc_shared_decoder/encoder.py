import torch.nn as nn
from .blocks import DownsampleBlock, ResidualBlock


class ImageEncoder(nn.Module):
    """ Encoder for extracting feature maps from input images."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            DownsampleBlock(3, 32, groups),
            DownsampleBlock(32, 64, groups),
            DownsampleBlock(64, 128, groups),
            DownsampleBlock(128, 256, groups),
            ResidualBlock(256, groups),
            ResidualBlock(256, groups),
        )

    def forward(self, x):
        return self.encoder(x)