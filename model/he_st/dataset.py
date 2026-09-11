import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class HESTDataset(Dataset):
    """Read HE-ST data from .pkl files and normalize ST data using provided statistics."""
    def __init__(self, pkl_dir: str, st_stats_path: str):
        self.pkl_paths = sorted(
            os.path.join(pkl_dir, name) for name in os.listdir(pkl_dir) if name.endswith(".pkl")
        )
        with np.load(st_stats_path) as stats:
            self.st_mean = stats["mean"].astype(np.float32).reshape(1, 1, -1)
            self.st_std = stats["safe_std"].astype(np.float32).reshape(1, 1, -1)

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, index):
        with open(self.pkl_paths[index], "rb") as f:
            data = pickle.load(f)

        he = torch.from_numpy(data["he_matrix_256"]).float()
        st = data["st_matrix"].astype(np.float32)
        st = torch.from_numpy((st - self.st_mean) / self.st_std)

        return {"he": he, "st": st, "sample_name": data["sample_name"]}