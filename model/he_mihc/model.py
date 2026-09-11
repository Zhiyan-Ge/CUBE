import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import ImageDecoder, MarkerSpecificMIHCDecoder


class DualReconstructionModel(nn.Module):
    """ A model that encodes an input image into a feature map (UR1) and then decodes it back to both HE and mIHC images."""
    def __init__(self, groups: int = 8, marker_specific_mihc: bool = False):
        super().__init__()
        self.encoder = ImageEncoder(groups)
        self.he_decoder = ImageDecoder(groups)

        if marker_specific_mihc:
            self.mihc_decoder = MarkerSpecificMIHCDecoder(groups)
        else:
            self.mihc_decoder = ImageDecoder(groups)

    def forward(self, x):
        ur1 = self.encoder(x)
        he_recon = self.he_decoder(ur1)
        mihc_recon = self.mihc_decoder(ur1)
        return ur1, he_recon, mihc_recon


class HEMIHCBranch(nn.Module):
    """Combines two DualReconstructionModels: one for HE input and one for mIHC input, 
       allowing for dual reconstruction and feature extraction."""
    def __init__(
        self,
        groups: int = 8,
        marker_specific_mihc: bool = True,
    ):
        super().__init__()

        self.model1 = DualReconstructionModel(
            groups,
            marker_specific_mihc=marker_specific_mihc,
        )

        self.model2 = DualReconstructionModel(
            groups,
            marker_specific_mihc=False,
        )

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