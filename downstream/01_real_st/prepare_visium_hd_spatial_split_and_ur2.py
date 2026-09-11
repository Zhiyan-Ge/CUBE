#!/usr/bin/env python3
"""Create spatial train/val/test splits and cache frozen CUBE UR2 features."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Edit each path independently when files move.
CUBE_MODEL_DIR = Path(
    "./model"
)
CUBE_CHECKPOINT = Path(
    "./"
    "models/test_models_3/test13/he_st/best.pt"
)
PATCH_METADATA = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_16um_paired/patch_metadata.csv"
)
NORMALIZED_PATCH_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_16um_macenko/patches_256"
)
OUT_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/prepared"
)


GPU_ID = 3
GROUPS = 8
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 2026
SPLIT_AXIS = "block_col"       # Vertical spatial stripes on the slide.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
GAP_COLUMNS = 1                # Unused stripe between adjacent splits.
SAVE_DTYPE = np.float16
SKIP_EXISTING = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_encoder(device: torch.device):
    sys.path.insert(0, str(CUBE_MODEL_DIR))
    from he_st.he_encoder import HEEncoder

    checkpoint = torch.load(CUBE_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    prefix = "model3.encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}

    encoder = HEEncoder(GROUPS)
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    return encoder, epoch


def make_spatial_split(metadata: pd.DataFrame) -> pd.DataFrame:
    coordinates = np.sort(metadata[SPLIT_AXIS].unique())
    usable = len(coordinates) - 2 * GAP_COLUMNS
    n_train = max(1, int(round(usable * TRAIN_RATIO)))
    n_val = max(1, int(round(usable * VAL_RATIO)))
    if n_train + n_val >= usable:
        n_val = max(1, usable - n_train - 1)

    train_values = coordinates[:n_train]
    gap1 = coordinates[n_train:n_train + GAP_COLUMNS]
    val_start = n_train + GAP_COLUMNS
    val_values = coordinates[val_start:val_start + n_val]
    gap2_start = val_start + n_val
    gap2 = coordinates[gap2_start:gap2_start + GAP_COLUMNS]
    test_values = coordinates[gap2_start + GAP_COLUMNS:]

    split = np.full(len(metadata), "unused", dtype=object)
    values = metadata[SPLIT_AXIS].to_numpy()
    split[np.isin(values, train_values)] = "train"
    split[np.isin(values, val_values)] = "val"
    split[np.isin(values, test_values)] = "test"

    output = metadata.copy()
    output["split"] = split
    output["split_axis"] = SPLIT_AXIS
    output["is_gap"] = np.isin(values, np.concatenate([gap1, gap2]))
    return output


def save_split_figure(metadata: pd.DataFrame, path: Path):
    colors = {"train": "#4daf4a", "val": "#ffb000", "test": "#377eb8", "unused": "#bdbdbd"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for split in ["train", "val", "test", "unused"]:
        part = metadata[metadata["split"] == split]
        ax.scatter(
            part["block_col"], part["block_row"], s=42,
            c=colors[split], label=f"{split} (n={len(part)})",
            edgecolors="black", linewidths=0.25,
        )
    ax.set_xlabel("block_col")
    ax.set_ylabel("block_row")
    ax.set_title("Visium HD spatial split for decoder-only fine-tuning")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


class PatchDataset(Dataset):
    def __init__(self, patch_ids: list[str]):
        self.patch_ids = patch_ids

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int):
        patch_id = self.patch_ids[index]
        image = cv2.imread(str(NORMALIZED_PATCH_DIR / f"{patch_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(image), patch_id


def cache_ur2(encoder, device: torch.device, patch_ids: list[str], feature_dir: Path):
    if SKIP_EXISTING:
        pending = [patch_id for patch_id in patch_ids if not (feature_dir / f"{patch_id}.npy").exists()]
    else:
        pending = patch_ids

    loader = DataLoader(
        PatchDataset(pending), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    with torch.inference_mode():
        for images, names in tqdm(loader, desc="Cache frozen UR2", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            features = encoder(images)
            features = features.float().cpu().numpy().astype(SAVE_DTYPE, copy=False)
            for name, feature in zip(names, features):
                np.save(feature_dir / f"{name}.npy", feature)
    return len(pending)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_dir = OUT_DIR / "ur2_features"
    feature_dir.mkdir(exist_ok=True)

    metadata = pd.read_csv(PATCH_METADATA).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    metadata = make_spatial_split(metadata)
    metadata.to_csv(OUT_DIR / "spatial_split.csv", index=False)
    save_split_figure(metadata, OUT_DIR / "spatial_split.png")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint_epoch = load_encoder(device)
    patch_ids = metadata["patch_id"].astype(str).tolist()
    newly_cached = cache_ur2(encoder, device, patch_ids, feature_dir)

    counts = metadata["split"].value_counts().to_dict()
    bins = metadata.groupby("split")["filtered_bins"].sum().astype(int).to_dict()
    report = {
        "method": "contiguous spatial stripes with one-block guard bands",
        "split_axis": SPLIT_AXIS,
        "train_ratio_before_guard_bands": TRAIN_RATIO,
        "val_ratio_before_guard_bands": VAL_RATIO,
        "gap_columns": GAP_COLUMNS,
        "patch_counts": {key: int(value) for key, value in counts.items()},
        "filtered_bin_counts": {key: int(value) for key, value in bins.items()},
        "cube_checkpoint": str(CUBE_CHECKPOINT),
        "checkpoint_epoch": checkpoint_epoch,
        "image_domain": "Macenko + global Lab L* normalized",
        "image_input_size": [256, 256],
        "ur2_shape": [256, 16, 16],
        "ur2_dtype": str(np.dtype(SAVE_DTYPE)),
        "features_newly_cached": newly_cached,
        "features_total": len(patch_ids),
    }
    with (OUT_DIR / "preparation_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD SPATIAL SPLIT AND UR2 CACHE COMPLETE")
    print("=" * 80)
    print(f"checkpoint epoch: {checkpoint_epoch}")
    print(f"train/val/test/unused: {counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}/{counts.get('unused', 0)}")
    print(f"new UR2 cached: {newly_cached}/{len(patch_ids)}")
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()
