import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and GroupNorm + GELU activation."""
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DownsampleBlock(nn.Module):
    """Downsample by 2 with a 3x3 convolution, followed by GroupNorm + GELU and a ResidualBlock."""
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(groups, out_channels), nn.GELU(),
            ResidualBlock(out_channels, groups)
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsampling by a factor of 2, followed by channel transformation and a ResidualBlock."""
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels), nn.GELU(),
            ResidualBlock(out_channels, groups)
        )

    def forward(self, x):
        return self.block(x)