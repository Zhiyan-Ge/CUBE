import torch.nn as nn
from .he_encoder import HEEncoder
from .st_encoder import STEncoder
from .he_decoder import HEDecoder
from .st_decoder import STDecoder


class Model3(nn.Module):
    """HE input model: generate HE source UR2, and independently reconstruct HE and ST."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.encoder = HEEncoder(groups)
        self.he_decoder = HEDecoder(groups)
        self.st_decoder = STDecoder(groups)

    def forward(self, he):
        ur2 = self.encoder(he)
        return ur2, self.he_decoder(ur2), self.st_decoder(ur2)


class Model4(nn.Module):
    """ST input model: generate ST source UR2, and independently reconstruct HE and ST."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.encoder = STEncoder(groups)
        self.he_decoder = HEDecoder(groups)
        self.st_decoder = STDecoder(groups)

    def forward(self, st):
        ur2 = self.encoder(st)
        return ur2, self.he_decoder(ur2), self.st_decoder(ur2)


class HESTBranch(nn.Module):
    """HE-ST branch model: generate HE source UR2 and ST source UR2, and independently reconstruct HE and ST."""
    def __init__(self, groups: int = 8):
        super().__init__()
        self.model3 = Model3(groups)
        self.model4 = Model4(groups)

    def forward(self, he, st):
        ur2_he, he_to_he, he_to_st = self.model3(he)
        ur2_st, st_to_he, st_to_st = self.model4(st)
        return {
            "ur2_he": ur2_he, "ur2_st": ur2_st,
            "he_to_he": he_to_he, "he_to_st": he_to_st,
            "st_to_he": st_to_he, "st_to_st": st_to_st
        }