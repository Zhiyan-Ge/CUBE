#!/usr/bin/env python3
"""Reconstruct HEMIT mIHC parent-region mosaics for CUBE Fig. 5B."""

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PKL_DIRS = {
    "train": Path("./data/train_data/final_data/train/pkl"),
    "val": Path("./data/train_data/final_data/val/pkl"),
    "test": Path("./data/train_data/final_data/test/pkl"),
}
OUT_DIR = Path("./result/03_spatial_cluster/result/fig5b_mihc_geometry")

PATCH_SIZE = 512
STRIDE = 256
FIGURE_DPI = 300


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def parse_sample_name(name):
    match = re.fullmatch(r"(.+)_patch_(\d+)_(\d+)", name)
    return match.group(1), int(match.group(2)), int(match.group(3))


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def load_records():
    rows = []
    for split, directory in PKL_DIRS.items():
        paths = sorted(directory.glob("*.pkl"))
        print(f"{split}: {len(paths)} PKLs")
        for path in paths:
            with path.open("rb") as f:
                data = pickle.load(f)
            name = str(data.get("sample_name", path.stem))
            mega_id, patch_i, patch_j = parse_sample_name(name)
            rows.append({
                "split": split, "sample_name": name, "mega_id": mega_id,
                "patch_i": patch_i, "patch_j": patch_j, "path": path
            })
    return pd.DataFrame(rows)


def load_mihc(path):
    with path.open("rb") as f:
        return to_numpy(pickle.load(f)["mihc_matrix_512"]).astype(np.float32)


def make_mosaic(group):
    n_rows = int(group["patch_i"].max()) + 1
    n_cols = int(group["patch_j"].max()) + 1
    height = PATCH_SIZE + (n_rows - 1) * STRIDE
    width = PATCH_SIZE + (n_cols - 1) * STRIDE
    canvas = np.zeros((3, height, width), dtype=np.float32)
    weight = np.zeros((1, height, width), dtype=np.float32)

    for row in group.itertuples():
        image = load_mihc(row.path)
        y, x = row.patch_i * STRIDE, row.patch_j * STRIDE
        canvas[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += image
        weight[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1

    canvas /= np.maximum(weight, 1)
    return np.clip(canvas, 0, 1)


def make_composite(mosaic):
    dapi, cd3, panck = mosaic
    composite = np.stack([panck, cd3, dapi], axis=-1)
    return np.clip(composite, 0, 1)


def save_panel(mosaic, composite, split, mega_id, path):
    dapi, cd3, panck = mosaic
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    axes[0].imshow(dapi, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("DAPI")
    axes[1].imshow(cd3, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("CD3")
    axes[2].imshow(panck, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("panCK")
    axes[3].imshow(composite)
    axes[3].set_title("Composite\npanCK=R, CD3=G, DAPI=B")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{split} | {mega_id}", fontsize=13, fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = {
        "DAPI": OUT_DIR / "DAPI",
        "CD3": OUT_DIR / "CD3",
        "panCK": OUT_DIR / "panCK",
        "composite": OUT_DIR / "composite",
        "panels": OUT_DIR / "panels",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    records = load_records()
    groups = list(records.groupby(["split", "mega_id"], sort=True))
    index_rows = []

    for index, ((split, mega_id), group) in enumerate(groups, start=1):
        mosaic = make_mosaic(group)
        composite = make_composite(mosaic)
        stem = f"{split}_{safe_name(mega_id)}"

        plt.imsave(dirs["DAPI"] / f"{stem}.png", mosaic[0], cmap="gray", vmin=0, vmax=1)
        plt.imsave(dirs["CD3"] / f"{stem}.png", mosaic[1], cmap="gray", vmin=0, vmax=1)
        plt.imsave(dirs["panCK"] / f"{stem}.png", mosaic[2], cmap="gray", vmin=0, vmax=1)
        plt.imsave(dirs["composite"] / f"{stem}.png", composite)
        save_panel(mosaic, composite, split, mega_id, dirs["panels"] / stem)

        index_rows.append({
            "split": split, "mega_id": mega_id, "n_patches": len(group),
            "grid_rows": int(group["patch_i"].max()) + 1,
            "grid_cols": int(group["patch_j"].max()) + 1,
            "DAPI": f"DAPI/{stem}.png", "CD3": f"CD3/{stem}.png",
            "panCK": f"panCK/{stem}.png", "composite": f"composite/{stem}.png",
            "panel": f"panels/{stem}.pdf"
        })
        print(f"[{index:02d}/{len(groups)}] {split} | {mega_id}")

    pd.DataFrame(index_rows).to_csv(OUT_DIR / "mihc_region_mosaic_index.csv", index=False)

    print("\n" + "=" * 80)
    print("mIHC REGION RECONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"Saved {len(groups)} regions to: {OUT_DIR}")


if __name__ == "__main__":
    main()
