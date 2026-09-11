import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import ImageDecoder


class DualReconstructionModel(nn.Module):
    """Single modal input, dual modal output. For example, input HE image, output HE and mIHC images."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.encoder = ImageEncoder(groups)
        self.he_decoder = ImageDecoder(groups)
        self.mihc_decoder = ImageDecoder(groups)

    def forward(self, x):
        ur1 = self.encoder(x)
        he_recon = self.he_decoder(ur1)
        mihc_recon = self.mihc_decoder(ur1)
        return ur1, he_recon, mihc_recon


class HEMIHCBranch(nn.Module):
    """Complete model with dual reconstruction branches. """
    def __init__(self, groups: int = 8):
        super().__init__()
        self.model1 = DualReconstructionModel(groups)  # HE -> HE / mIHC
        self.model2 = DualReconstructionModel(groups)  # mIHC -> HE / mIHC

    def forward(self, he, mihc):
        ur1_he, he_to_he, he_to_mihc = self.model1(he)
        ur1_mihc, mihc_to_he, mihc_to_mihc = self.model2(mihc)

        return {
            "ur1_he": ur1_he,
            "ur1_mihc": ur1_mihc,
            "he_to_he": he_to_he,
            "he_to_mihc": he_to_mihc,
            "mihc_to_he": mihc_to_he,
            "mihc_to_mihc": mihc_to_mihc,
        }