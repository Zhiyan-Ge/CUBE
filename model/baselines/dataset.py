import os
import pickle

import torch
from torch.utils.data import Dataset


class HEMITPKLDataset(Dataset):
    """Read CUBE 512×512 paired HE/mIHC PKLs and map [0,1] to [-1,1]."""

    def __init__(self, pkl_dir: str):
        self.pkl_paths = sorted(
            os.path.join(pkl_dir, name)
            for name in os.listdir(pkl_dir)
            if name.endswith(".pkl")
        )
        if not self.pkl_paths:
            raise RuntimeError(f"No .pkl files found in: {pkl_dir}")

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, index):
        path = self.pkl_paths[index]
        with open(path, "rb") as f:
            data = pickle.load(f)

        he = torch.from_numpy(data["he_matrix_512"]).float()
        mihc = torch.from_numpy(data["mihc_matrix_512"]).float()

        # Supplied HEMIT/pix2pix generators end in Tanh and were trained on [-1,1].
        he = he.mul(2.0).sub(1.0)
        mihc = mihc.mul(2.0).sub(1.0)

        return {
            "he": he,
            "mihc": mihc,
            "sample_name": data["sample_name"],
        }
