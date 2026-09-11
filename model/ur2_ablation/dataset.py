import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class UR2Dataset(Dataset):
    def __init__(self, pkl_dir, ur_dir):
        self.pkl_dir, self.ur_dir = pkl_dir, ur_dir
        def names(path, ext):
            if not os.path.isdir(path): raise FileNotFoundError(path)
            return {os.path.splitext(x)[0] for x in os.listdir(path) if x.endswith(ext)}
        pkl_names, ur_names = names(pkl_dir, ".pkl"), names(ur_dir, ".npy")
        if pkl_names != ur_names:
            raise RuntimeError(f"PKL / UR2 mismatch: PKL={len(pkl_names)}, UR2={len(ur_names)}\nMissing UR2={sorted(pkl_names-ur_names)[:5]}\nExtra UR2={sorted(ur_names-pkl_names)[:5]}")
        self.sample_names = sorted(pkl_names)
        if not self.sample_names: raise RuntimeError("No matched samples found")
        x = np.load(os.path.join(ur_dir, self.sample_names[0] + ".npy"), mmap_mode="r")
        if x.shape != (256,16,16): raise ValueError(f"UR2 shape={x.shape}, expected (256,16,16)")
        print(f"UR2Dataset ready | samples={len(self.sample_names)} | UR2={x.shape}")

    def __len__(self): return len(self.sample_names)

    def __getitem__(self, i):
        name = self.sample_names[i]
        ur = torch.from_numpy(np.load(os.path.join(self.ur_dir, name + ".npy"))).float()
        with open(os.path.join(self.pkl_dir, name + ".pkl"), "rb") as f: d = pickle.load(f)
        coord = torch.as_tensor(d["normalized_mega_coord"], dtype=torch.float32).reshape(-1)
        concept = torch.as_tensor(d["concept_scores"], dtype=torch.float32).reshape(-1)
        if coord.numel()!=2 or concept.numel()!=3: raise ValueError(f"{name}: coord={tuple(coord.shape)}, concept={tuple(concept.shape)}")
        return {"ur":ur,"coord":coord,"concept":concept,"sample_name":name}
