#!/usr/bin/env python3
"""Export frozen CUBE UR1 features for Fig. 5D Visium HD patches."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CUBE_MODEL_DIR = Path("./model")
CHECKPOINT = Path("./models/test_models_3/test11/he_mihc/best_pearson.pt")
SPATIAL_SPLIT = Path("./result/01_real_st/data/HD/visium_hd_cube_finetune/prepared/spatial_split.csv")
PATCH_DIR = Path("./result/01_real_st/data/HD/visium_hd_16um_macenko/patches_512")
OUT_DIR = Path("./result/05_real_st_program/data/visium_hd_features/ur1")

GPU_ID = 3
GROUPS = 8
BATCH_SIZE = 32
NUM_WORKERS = 8
SAVE_DTYPE = np.float32

sys.path.insert(0, str(CUBE_MODEL_DIR))
from he_mihc.encoder import ImageEncoder


class PatchDataset(Dataset):
    def __init__(self, patch_ids):
        self.patch_ids = patch_ids

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]
        image = cv2.imread(str(PATCH_DIR / f"{patch_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(image), patch_id


def load_encoder(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    prefix = "model1.encoder."
    encoder = ImageEncoder(GROUPS)
    encoder.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    return encoder.to(device).eval(), checkpoint.get("epoch", -1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SPATIAL_SPLIT).sort_values(["block_col", "block_row"])
    patch_ids = metadata["patch_id"].astype(str).tolist()
    loader = DataLoader(
        PatchDataset(patch_ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    encoder, epoch = load_encoder(device)
    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        for images, names in tqdm(loader, desc="Export Visium HD UR1", dynamic_ncols=True):
            features = encoder(images.to(device, non_blocking=True))
            features = features.float().cpu().numpy().astype(SAVE_DTYPE, copy=False)
            for name, feature in zip(names, features):
                np.save(OUT_DIR / f"{name}.npy", feature)

    print(f"checkpoint epoch: {epoch}")
    print(f"exported UR1: {len(patch_ids)} patches")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()