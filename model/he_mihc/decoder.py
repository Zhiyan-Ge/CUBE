import torch
import torch.nn as nn
from .blocks import UpsampleBlock


class ImageDecoder(nn.Module):
    """Decoder for reconstructing images from feature maps."""
    def __init__(self, groups: int = 8, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            UpsampleBlock(256, 128, groups),
            UpsampleBlock(128, 64, groups),
            UpsampleBlock(64, 32, groups),
            UpsampleBlock(32, 32, groups),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, ur1):
        return self.decoder(ur1)


class MarkerSpecificMIHCDecoder(nn.Module):
    """ Decoder for reconstructing marker-specific mIHC images from UR1 feature maps."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.dapi_decoder = ImageDecoder(groups, out_channels=1)
        self.cd3_decoder = ImageDecoder(groups, out_channels=1)
        self.panck_decoder = ImageDecoder(groups, out_channels=1)

    def forward(self, ur1):
        dapi = self.dapi_decoder(ur1)
        cd3 = self.cd3_decoder(ur1)
        panck = self.panck_decoder(ur1)

        return torch.cat([dapi, cd3, panck], dim=1)