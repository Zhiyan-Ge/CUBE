#!/usr/bin/env python3
"""Export coordinate-ablated CUBE fusion using one fixed coordinate for all patches."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# Edit only this section if files move.
CUBE_MODEL_DIR = Path("./model")
FUSION_CHECKPOINT = Path(
    "./models/test_models_3/test15/ur_branch/best.pt"
)
PKL_DIRS = {
    "train": Path("./data/train_data/final_data/train/pkl"),
    "val": Path("./data/train_data/final_data/val/pkl"),
    "test": Path("./data/train_data/final_data/test/pkl"),
}
UR1_DIRS = {
    split: Path("./models/test_models_3/test11/he_mihc/ur1_he") / split
    for split in ("train", "val", "test")
}
UR2_DIRS = {
    split: Path("./models/test_models_3/test13/he_st/ur2_he") / split
    for split in ("train", "val", "test")
}
OUT_DIR = Path(
    "./result/02_UMAP/data/fused_ur_fixed_coord"
)


SPLITS = ("train", "val", "test")
GPU_ID = 3
BATCH_SIZE = 4
NUM_WORKERS = 8
DIM = 256
HEADS = 8
FFN_DIM = 512
DROPOUT = 0.1
RANDOM_STATE = 2026
SAVE_DTYPE = np.float32
SKIP_EXISTING = True
FIXED_COORD_SOURCE = "all_splits_mean"
MANIFEST_NAME = "fixed_coord_export_manifest.json"


sys.path.insert(0, str(CUBE_MODEL_DIR))
from ur.model import URModel


def stems(path, suffix):
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    suffix = suffix.lower()
    return {file.stem for file in path.iterdir() if file.is_file() and file.suffix.lower() == suffix}


def load_coord(path):
    with path.open("rb") as file:
        payload = pickle.load(file)
    if "normalized_mega_coord" not in payload:
        raise RuntimeError(f"Missing normalized_mega_coord: {path}")
    value = payload["normalized_mega_coord"]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    coord = np.asarray(value, dtype=np.float64).reshape(-1)
    if coord.shape != (2,) or not np.isfinite(coord).all():
        raise RuntimeError(f"Invalid normalized_mega_coord in {path}: shape={coord.shape}, value={coord}")
    return coord


def validate_input_names(split):
    pkl_names = stems(PKL_DIRS[split], ".pkl")
    ur1_names = stems(UR1_DIRS[split], ".npy")
    ur2_names = stems(UR2_DIRS[split], ".npy")
    if pkl_names != ur1_names or pkl_names != ur2_names:
        lines = [
            f"Input sample mismatch in {split}:",
            f"  PKL={len(pkl_names)} | UR1={len(ur1_names)} | UR2={len(ur2_names)}",
        ]
        for names, label in ((ur1_names, "UR1"), (ur2_names, "UR2")):
            missing = sorted(pkl_names - names)
            extra = sorted(names - pkl_names)
            if missing:
                lines.append(f"  {label} missing examples: {missing[:10]}")
            if extra:
                lines.append(f"  {label} extra examples: {extra[:10]}")
        raise RuntimeError("\n".join(lines))
    if not pkl_names:
        raise RuntimeError(f"No samples found in split: {split}")
    return sorted(pkl_names)


def compute_fixed_coordinate(sample_names):
    coordinates, split_counts = [], {}
    for split in SPLITS:
        names = sample_names[split]
        split_counts[split] = len(names)
        print(f"Reading coordinates from {split}: {len(names)} patches")
        coordinates.extend(load_coord(PKL_DIRS[split] / f"{name}.pkl") for name in names)
    matrix = np.stack(coordinates, axis=0)
    fixed = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    unique_coordinates = np.unique(np.round(matrix, decimals=8), axis=0)
    statistics = {
        "source": FIXED_COORD_SOURCE,
        "n_patches": int(len(matrix)),
        "n_unique_coordinates": int(len(unique_coordinates)),
        "split_counts": split_counts,
        "fixed_coordinate": fixed.astype(float).tolist(),
        "coordinate_min": matrix.min(axis=0).astype(float).tolist(),
        "coordinate_max": matrix.max(axis=0).astype(float).tolist(),
        "coordinate_std": matrix.std(axis=0).astype(float).tolist(),
    }
    return fixed, statistics


class FixedCoordinateDataset(Dataset):
    def __init__(self, split, sample_names):
        self.split = split
        self.sample_names = sample_names

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        ur1 = np.asarray(np.load(UR1_DIRS[self.split] / f"{name}.npy", allow_pickle=False), dtype=np.float32)
        ur2 = np.asarray(np.load(UR2_DIRS[self.split] / f"{name}.npy", allow_pickle=False), dtype=np.float32)
        if ur1.shape != (256, 32, 32):
            raise RuntimeError(f"Expected UR1 (256,32,32), got {ur1.shape}: {name}")
        if ur2.shape != (256, 16, 16):
            raise RuntimeError(f"Expected UR2 (256,16,16), got {ur2.shape}: {name}")
        if not np.isfinite(ur1).all() or not np.isfinite(ur2).all():
            raise RuntimeError(f"Non-finite UR feature: {name}")
        return {
            "ur1": torch.from_numpy(ur1),
            "ur2": torch.from_numpy(ur2),
            "sample_name": name,
        }


def load_model(device):
    if not CUBE_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"CUBE model directory not found: {CUBE_MODEL_DIR}")
    if not FUSION_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {FUSION_CHECKPOINT}")
    checkpoint = torch.load(FUSION_CHECKPOINT, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model = URModel(DIM, HEADS, FFN_DIM, DROPOUT)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    return model, epoch


def checkpoint_identity(epoch):
    stat = FUSION_CHECKPOINT.stat()
    return {
        "path": str(FUSION_CHECKPOINT),
        "epoch": epoch,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_manifest(manifest):
    path = OUT_DIR / MANIFEST_NAME
    temporary = OUT_DIR / f"{MANIFEST_NAME}.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def check_existing_manifest(fixed_coord, checkpoint_info):
    path = OUT_DIR / MANIFEST_NAME
    existing_outputs = any((OUT_DIR / split).is_dir() and any((OUT_DIR / split).glob("*.npy")) for split in SPLITS)
    if not path.is_file():
        if existing_outputs:
            raise RuntimeError(
                f"Existing NPY files found under {OUT_DIR}, but {MANIFEST_NAME} is missing. "
                "Use a new OUT_DIR to avoid mixing incompatible exports."
            )
        return
    with path.open(encoding="utf-8") as file:
        previous = json.load(file)
    previous_coord = np.asarray(previous.get("coordinate", {}).get("fixed_coordinate", []), dtype=np.float64)
    same_coord = previous_coord.shape == (2,) and np.allclose(previous_coord, fixed_coord, rtol=0, atol=1e-7)
    previous_checkpoint = previous.get("checkpoint", {})
    same_checkpoint = (
        previous_checkpoint.get("path") == checkpoint_info["path"]
        and previous_checkpoint.get("epoch") == checkpoint_info["epoch"]
        and previous_checkpoint.get("size_bytes") == checkpoint_info["size_bytes"]
    )
    if not same_coord or not same_checkpoint:
        raise RuntimeError(
            f"Existing export manifest is incompatible with the current fixed coordinate or checkpoint: {path}\n"
            "Use a new OUT_DIR instead of mixing two coordinate-ablation exports."
        )


def validate_saved_feature(path):
    feature = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if feature.shape != (512,) or not np.isfinite(feature).all():
        raise RuntimeError(f"Invalid saved fixed-coordinate fusion: {path}, shape={feature.shape}")


def export_split(model, device, fixed_coord, split, names):
    dataset = FixedCoordinateDataset(split, names)
    split_out = OUT_DIR / split
    split_out.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(dataset)))
    if SKIP_EXISTING:
        kept = []
        for index, name in enumerate(dataset.sample_names):
            path = split_out / f"{name}.npy"
            if path.exists():
                validate_saved_feature(path)
            else:
                kept.append(index)
        indices = kept
    if indices:
        loader = DataLoader(
            Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        fixed = torch.from_numpy(fixed_coord).to(device)
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Export fixed-coordinate fusion {split}", dynamic_ncols=True):
                ur1 = batch["ur1"].to(device, non_blocking=True)
                ur2 = batch["ur2"].to(device, non_blocking=True)
                coord = fixed.unsqueeze(0).expand(ur1.shape[0], -1)
                fused = model.encode(ur1, ur2, coord)
                if fused.ndim != 2 or fused.shape[1] != 512 or not torch.isfinite(fused).all():
                    raise RuntimeError(f"Invalid model.encode output shape/value in {split}: {tuple(fused.shape)}")
                fused = fused.float().cpu().numpy().astype(SAVE_DTYPE, copy=False)
                for name, feature in zip(batch["sample_name"], fused):
                    np.save(split_out / f"{name}.npy", feature)

    output_names = stems(split_out, ".npy")
    expected_names = set(names)
    if output_names != expected_names:
        missing = sorted(expected_names - output_names)
        extra = sorted(output_names - expected_names)
        raise RuntimeError(
            f"Output mismatch in {split}: expected={len(expected_names)}, observed={len(output_names)}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for name in names:
        validate_saved_feature(split_out / f"{name}.npy")
    print(f"{split}: exported_now={len(indices)} | total={len(names)} | output={split_out}")
    return len(indices)


def main():
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_names = {split: validate_input_names(split) for split in SPLITS}
    fixed_coord, coordinate_info = compute_fixed_coordinate(sample_names)
    model, epoch = load_model(device)
    checkpoint_info = checkpoint_identity(epoch)
    check_existing_manifest(fixed_coord, checkpoint_info)
    manifest = {
        "status": "running",
        "analysis": "CUBE fixed-coordinate fusion export",
        "method": "URModel.encode with one all-splits mean coordinate shared by every patch",
        "coordinate": coordinate_info,
        "checkpoint": checkpoint_info,
        "output_directory": str(OUT_DIR),
        "output_dtype": str(np.dtype(SAVE_DTYPE)),
        "expected_dimension": 512,
        "exported_now": {},
    }
    write_manifest(manifest)

    print(f"Loaded checkpoint: {FUSION_CHECKPOINT} | epoch={epoch} | device={device}")
    print(f"Fixed coordinate used for every patch: {fixed_coord.tolist()}")
    for split in SPLITS:
        manifest["exported_now"][split] = export_split(
            model, device, fixed_coord, split, sample_names[split]
        )
        write_manifest(manifest)
    manifest["status"] = "completed"
    manifest["final_counts"] = {split: len(sample_names[split]) for split in SPLITS}
    write_manifest(manifest)
    print("Fixed-coordinate fused UR export finished.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
