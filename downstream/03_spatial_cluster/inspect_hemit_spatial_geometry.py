#!/usr/bin/env python3
"""Inspect HEMIT parent-region spatial geometry for CUBE Fig. 5B."""

import json
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
OUT_DIR = Path("./result/03_spatial_cluster/result/fig5b_spatial_geometry")

PATCH_SIZE = 512
STRIDE = 256
OVERLAP = PATCH_SIZE - STRIDE
N_EXAMPLES = 5
RANDOM_STATE = 2026
FIGURE_DPI = 300


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def parse_mega_id(sample_name):
    return re.sub(r"_patch_\d+_\d+$", "", sample_name)


def load_records():
    rows = []
    for split, directory in PKL_DIRS.items():
        paths = sorted(directory.glob("*.pkl"))
        print(f"{split}: {len(paths)} PKLs")
        for path in paths:
            with path.open("rb") as f:
                data = pickle.load(f)
            sample_name = str(data.get("sample_name", path.stem))
            grid = to_numpy(data["patch_grid_coord"]).astype(int).reshape(-1)
            rows.append({
                "split": split, "sample_name": sample_name, "mega_id": parse_mega_id(sample_name),
                "patch_i": int(grid[0]), "patch_j": int(grid[1]), "path": path
            })
    return pd.DataFrame(rows)


def load_he(path):
    with path.open("rb") as f:
        image = to_numpy(pickle.load(f)["he_matrix_512"]).astype(np.float32)
    return image


def build_region_table(records):
    rows = []
    for (split, mega_id), group in records.groupby(["split", "mega_id"], sort=True):
        i_values = sorted(group["patch_i"].unique())
        j_values = sorted(group["patch_j"].unique())
        observed = set(zip(group["patch_i"], group["patch_j"]))
        expected = {(i, j) for i in i_values for j in j_values}
        rows.append({
            "split": split, "mega_id": mega_id, "n_patches": len(group),
            "patch_i_min": min(i_values), "patch_i_max": max(i_values), "patch_i_unique": len(i_values),
            "patch_j_min": min(j_values), "patch_j_max": max(j_values), "patch_j_unique": len(j_values),
            "complete_rectangular_grid": observed == expected
        })
    return pd.DataFrame(rows)


def candidates():
    out = []
    for swap in (False, True):
        for flip_row in (False, True):
            for flip_col in (False, True):
                out.append({
                    "name": f"{'row=j,col=i' if swap else 'row=i,col=j'}|row_{'rev' if flip_row else 'fwd'}|col_{'rev' if flip_col else 'fwd'}",
                    "swap": swap, "flip_row": flip_row, "flip_col": flip_col
                })
    return out


def transform(i, j, candidate, i_min, i_max, j_min, j_max):
    if candidate["swap"]:
        row = j_max - j if candidate["flip_row"] else j - j_min
        col = i_max - i if candidate["flip_col"] else i - i_min
    else:
        row = i_max - i if candidate["flip_row"] else i - i_min
        col = j_max - j if candidate["flip_col"] else j - j_min
    return int(row), int(col)


def overlap_mae(a, b, direction):
    if direction == "right":
        return float(np.mean(np.abs(a[:, :, STRIDE:] - b[:, :, :OVERLAP])))
    return float(np.mean(np.abs(a[:, STRIDE:, :] - b[:, :OVERLAP, :])))


def score_orientations(records):
    result = []
    region_groups = list(records.groupby(["split", "mega_id"], sort=True))
    for candidate in candidates():
        errors, right_errors, down_errors = [], [], []
        for _, group in region_groups:
            i_values, j_values = group["patch_i"].unique(), group["patch_j"].unique()
            i_min, i_max, j_min, j_max = min(i_values), max(i_values), min(j_values), max(j_values)
            positioned = {}
            for row in group.itertuples():
                pos = transform(row.patch_i, row.patch_j, candidate, i_min, i_max, j_min, j_max)
                positioned[pos] = load_he(row.path)
            for (r, c), image in positioned.items():
                if (r, c + 1) in positioned:
                    e = overlap_mae(image, positioned[(r, c + 1)], "right")
                    errors.append(e)
                    right_errors.append(e)
                if (r + 1, c) in positioned:
                    e = overlap_mae(image, positioned[(r + 1, c)], "down")
                    errors.append(e)
                    down_errors.append(e)
        result.append({
            **candidate,
            "mean_overlap_mae": float(np.mean(errors)),
            "right_overlap_mae": float(np.mean(right_errors)),
            "down_overlap_mae": float(np.mean(down_errors)),
            "n_neighbor_pairs": len(errors)
        })
    return pd.DataFrame(result).sort_values("mean_overlap_mae").reset_index(drop=True)


def make_mosaic(group, candidate):
    i_values, j_values = group["patch_i"].unique(), group["patch_j"].unique()
    i_min, i_max, j_min, j_max = min(i_values), max(i_values), min(j_values), max(j_values)
    tiles = []
    for row in group.itertuples():
        r, c = transform(row.patch_i, row.patch_j, candidate, i_min, i_max, j_min, j_max)
        tiles.append((r, c, load_he(row.path)))
    n_rows = max(r for r, _, _ in tiles) + 1
    n_cols = max(c for _, c, _ in tiles) + 1
    height = PATCH_SIZE + (n_rows - 1) * STRIDE
    width = PATCH_SIZE + (n_cols - 1) * STRIDE
    canvas = np.zeros((3, height, width), dtype=np.float32)
    weight = np.zeros((1, height, width), dtype=np.float32)
    for r, c, image in tiles:
        y, x = r * STRIDE, c * STRIDE
        canvas[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += image
        weight[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1
    canvas /= np.maximum(weight, 1)
    return np.clip(canvas.transpose(1, 2, 0), 0, 1), n_rows, n_cols


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mosaic_dir = OUT_DIR / "mosaics"
    all_region_dir = mosaic_dir / "all_regions"
    all_region_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    region_table = build_region_table(records)
    region_table.to_csv(OUT_DIR / "parent_region_geometry.csv", index=False)

    shape_counts = region_table.groupby(
        ["patch_i_unique", "patch_j_unique", "n_patches", "complete_rectangular_grid"]
    ).size().reset_index(name="n_regions").sort_values("n_regions", ascending=False)
    shape_counts.to_csv(OUT_DIR / "parent_region_shape_counts.csv", index=False)

    print("\nParent-region geometry")
    print("=" * 80)
    print(f"Total patches: {len(records)}")
    print(f"Parent regions: {len(region_table)}")
    print(shape_counts.to_string(index=False))

    scores = score_orientations(records)
    scores.to_csv(OUT_DIR / "orientation_overlap_scores.csv", index=False, float_format="%.8g")
    best = scores.iloc[0]
    candidate = {
        "name": best["name"], "swap": bool(best["swap"]),
        "flip_row": bool(best["flip_row"]), "flip_col": bool(best["flip_col"])
    }

    print("\nOrientation scores")
    print("=" * 80)
    print(scores[["name", "mean_overlap_mae", "right_overlap_mae", "down_overlap_mae"]].to_string(index=False))
    print(f"\nSelected orientation: {candidate['name']}")

    # Save all 84 reconstructed H&E parent-region mosaics.
    region_keys = list(records.groupby(["split", "mega_id"], sort=True).groups.keys())
    mosaics = {}
    mosaic_index = []
    print("\nSaving all parent-region mosaics...")
    for index, (split, mega_id) in enumerate(region_keys, start=1):
        group = records[(records["split"] == split) & (records["mega_id"] == mega_id)]
        image, n_rows, n_cols = make_mosaic(group, candidate)
        filename = f"{split}_{safe_name(mega_id)}.png"
        plt.imsave(all_region_dir / filename, image)
        mosaics[(split, mega_id)] = image
        mosaic_index.append({
            "split": split, "mega_id": mega_id, "n_rows": n_rows, "n_cols": n_cols,
            "n_patches": len(group), "filename": filename
        })
        print(f"[{index:02d}/{len(region_keys)}] {split} | {mega_id} -> {filename}")

    pd.DataFrame(mosaic_index).to_csv(OUT_DIR / "parent_region_mosaic_index.csv", index=False)

    # Keep five random examples as a compact geometry overview.
    rng = np.random.default_rng(RANDOM_STATE)
    selected = [region_keys[i] for i in rng.choice(len(region_keys), min(N_EXAMPLES, len(region_keys)), replace=False)]
    examples = []
    index_lookup = {(item["split"], item["mega_id"]): item for item in mosaic_index}
    for split, mega_id in selected:
        item = index_lookup[(split, mega_id)]
        examples.append({**item, "image": mosaics[(split, mega_id)]})

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 4 * len(examples)))
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, examples):
        ax.imshow(item["image"])
        ax.set_title(f"{item['split']} | {item['mega_id']} | grid={item['n_rows']}x{item['n_cols']}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "example_parent_region_mosaics.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "example_parent_region_mosaics.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_patches": int(len(records)),
        "n_parent_regions": int(len(region_table)),
        "selected_orientation": {
            **candidate,
            "mean_overlap_mae": float(best["mean_overlap_mae"]),
            "right_overlap_mae": float(best["right_overlap_mae"]),
            "down_overlap_mae": float(best["down_overlap_mae"])
        },
        "working_geometry": {"patch_size": PATCH_SIZE, "stride": STRIDE, "overlap": OVERLAP},
        "all_region_mosaic_directory": str(all_region_dir),
        "n_saved_mosaics": int(len(mosaic_index)),
        "example_regions": [{k: v for k, v in item.items() if k != "image"} for item in examples]
    }
    with (OUT_DIR / "spatial_geometry_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved {len(mosaic_index)} parent-region mosaics to: {all_region_dir}")
    print(f"Mosaic index: {OUT_DIR / 'parent_region_mosaic_index.csv'}")
    print(f"Examples: {OUT_DIR / 'example_parent_region_mosaics.pdf'}")


if __name__ == "__main__":
    main()
