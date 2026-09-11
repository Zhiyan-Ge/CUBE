import os
import pickle
import torch
from torch.utils.data import Dataset


class HEMIHCDataset(Dataset):
    """Dataset for loading HE and mIHC image pairs from .pkl files."""
    def __init__(self, pkl_dir: str):
        self.pkl_paths = sorted(
            os.path.join(pkl_dir, name) for name in os.listdir(pkl_dir) if name.endswith(".pkl")
        )

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, index):
        with open(self.pkl_paths[index], "rb") as f:
            data = pickle.load(f)

        he = torch.from_numpy(data["he_matrix_512"]).float()
        mihc = torch.from_numpy(data["mihc_matrix_512"]).float()

        return {
            "he": he,
            "mihc": mihc,
            "sample_name": data["sample_name"],
        }