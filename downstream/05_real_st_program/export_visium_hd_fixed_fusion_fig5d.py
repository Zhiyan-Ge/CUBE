#!/usr/bin/env python3
"""Export fixed-coordinate CUBE Fusion features for Fig. 5D Visium HD patches."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CUBE_MODEL_DIR = Path("./model")
FUSION_CHECKPOINT = Path("./models/test_models_3/test15/ur_branch/best.pt")
FIXED_COORD_MANIFEST = Path("./result/02_UMAP/data/fused_ur_fixed_coord/fixed_coord_export_manifest.json")
SPATIAL_SPLIT = Path("./result/01_real_st/data/HD/visium_hd_cube_finetune/prepared/spatial_split.csv")

UR1_DIR = Path("./result/05_real_st_program/data/visium_hd_features/ur1")
UR2_DIR = Path("./result/01_real_st/data/HD/visium_hd_cube_finetune/prepared/ur2_features")
OUT_DIR = Path("./result/05_real_st_program/data/visium_hd_features/fusion_fixed_coord")

GPU_ID = 3
BATCH_SIZE = 4
NUM_WORKERS = 8
DIM = 256
HEADS = 8
FFN_DIM = 512
DROPOUT = 0.1
SAVE_DTYPE = np.float32

sys.path.insert(0, str(CUBE_MODEL_DIR))
from ur.model import URModel


class FeatureDataset(Dataset):
    def __init__(self, patch_ids):
        self.patch_ids = patch_ids

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]
        ur1 = np.load(UR1_DIR / f"{patch_id}.npy").astype(np.float32)
        ur2 = np.load(UR2_DIR / f"{patch_id}.npy").astype(np.float32)
        return torch.from_numpy(ur1), torch.from_numpy(ur2), patch_id


def load_model(device):
    checkpoint = torch.load(FUSION_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model = URModel(DIM, HEADS, FFN_DIM, DROPOUT)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint.get("epoch", -1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SPATIAL_SPLIT).sort_values(["block_col", "block_row"])
    patch_ids = metadata["patch_id"].astype(str).tolist()
    loader = DataLoader(
        FeatureDataset(patch_ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    with FIXED_COORD_MANIFEST.open(encoding="utf-8") as file:
        fixed_coord = np.asarray(
            json.load(file)["coordinate"]["fixed_coordinate"], dtype=np.float32
        )

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(device)
    fixed_coord = torch.from_numpy(fixed_coord).to(device)

    with torch.inference_mode():
        for ur1, ur2, names in tqdm(loader, desc="Export Visium HD fixed Fusion", dynamic_ncols=True):
            ur1 = ur1.to(device, non_blocking=True)
            ur2 = ur2.to(device, non_blocking=True)
            coord = fixed_coord.unsqueeze(0).expand(ur1.size(0), -1)
            features = model.encode(ur1, ur2, coord)
            features = features.float().cpu().numpy().astype(SAVE_DTYPE, copy=False)
            for name, feature in zip(names, features):
                np.save(OUT_DIR / f"{name}.npy", feature)

    print(f"checkpoint epoch: {epoch}")
    print(f"fixed coordinate: {fixed_coord.cpu().numpy().tolist()}")
    print(f"exported Fusion: {len(patch_ids)} patches")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()