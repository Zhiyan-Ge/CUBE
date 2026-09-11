import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class URDataset(Dataset):
    """Read UR1/UR2 .npy and corresponding .pkl files, return a dict with tensors."""
    def __init__(self, pkl_dir: str, ur1_dir: str, ur2_dir: str):
        self.pkl_dir, self.ur1_dir, self.ur2_dir = pkl_dir, ur1_dir, ur2_dir

        def names(path, ext):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Directory not found: {path}")
            return {os.path.splitext(x)[0] for x in os.listdir(path) if x.endswith(ext)}

        pkl_names = names(pkl_dir, ".pkl")
        ur1_names = names(ur1_dir, ".npy")
        ur2_names = names(ur2_dir, ".npy")

        if pkl_names != ur1_names or pkl_names != ur2_names:
            missing_ur1 = sorted(pkl_names - ur1_names)
            missing_ur2 = sorted(pkl_names - ur2_names)
            extra_ur1 = sorted(ur1_names - pkl_names)
            extra_ur2 = sorted(ur2_names - pkl_names)
            raise RuntimeError(
                "PKL / UR1 / UR2 sample mismatch!\n"
                f"PKL={len(pkl_names)}, UR1={len(ur1_names)}, UR2={len(ur2_names)}\n"
                f"Missing UR1: {missing_ur1[:5]}\n"
                f"Missing UR2: {missing_ur2[:5]}\n"
                f"Extra UR1: {extra_ur1[:5]}\n"
                f"Extra UR2: {extra_ur2[:5]}"
            )

        self.sample_names = sorted(pkl_names)
        if not self.sample_names:
            raise RuntimeError("No matched samples found.")

        # Check the first sample for shape and keys
        name = self.sample_names[0]
        ur1 = np.load(os.path.join(ur1_dir, name + ".npy"), mmap_mode="r")
        ur2 = np.load(os.path.join(ur2_dir, name + ".npy"), mmap_mode="r")
        if ur1.shape != (256, 32, 32):
            raise ValueError(f"UR1 shape error: {ur1.shape}, expected (256,32,32)")
        if ur2.shape != (256, 16, 16):
            raise ValueError(f"UR2 shape error: {ur2.shape}, expected (256,16,16)")

        with open(os.path.join(pkl_dir, name + ".pkl"), "rb") as f:
            data = pickle.load(f)
        for key in ("normalized_mega_coord", "concept_scores"):
            if key not in data:
                raise KeyError(f"{name}.pkl missing key: {key}")

        print(f"URDataset ready | samples={len(self.sample_names)}")
        print(f"UR1={ur1.shape} | UR2={ur2.shape}")

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        ur1 = torch.from_numpy(np.load(os.path.join(self.ur1_dir, name + ".npy"))).float()
        ur2 = torch.from_numpy(np.load(os.path.join(self.ur2_dir, name + ".npy"))).float()

        with open(os.path.join(self.pkl_dir, name + ".pkl"), "rb") as f:
            data = pickle.load(f)

        coord = torch.as_tensor(data["normalized_mega_coord"], dtype=torch.float32).reshape(-1)
        concept = torch.as_tensor(data["concept_scores"], dtype=torch.float32).reshape(-1)

        if coord.numel() != 2:
            raise ValueError(f"{name}: coord shape={tuple(coord.shape)}, expected 2 values")
        if concept.numel() != 3:
            raise ValueError(f"{name}: concept shape={tuple(concept.shape)}, expected 3 values")

        return {
            "ur1": ur1,
            "ur2": ur2,
            "coord": coord,
            "concept": concept,
            "sample_name": name
        }