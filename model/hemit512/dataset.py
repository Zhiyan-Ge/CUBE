import os
import pickle

import torch
from torch.utils.data import Dataset


class HEMITPKLDataset(Dataset):
    """Read CUBE 512×512 paired H&E/mIHC PKLs and map [0,1] -> [-1,1]."""

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

        if he.shape != (3, 512, 512):
            raise ValueError(f"Unexpected HE shape {tuple(he.shape)} in {path}")
        if mihc.shape != (3, 512, 512):
            raise ValueError(f"Unexpected mIHC shape {tuple(mihc.shape)} in {path}")

        he = he.mul(2.0).sub(1.0)
        mihc = mihc.mul(2.0).sub(1.0)

        return {
            "he": he,
            "mihc": mihc,
            "sample_name": data.get("sample_name", os.path.basename(path)),
        }
