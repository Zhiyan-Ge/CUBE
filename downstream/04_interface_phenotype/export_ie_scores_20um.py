#!/usr/bin/env python3
"""Export final 20-um immune-epithelial spatial enrichment scores for CUBE Fig. 5C."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DATA_ROOT = Path("./data/paired_data/he_mihc/raw_data/HEMIT_dataset")
OUT_DIR = Path("./result/04_interface_phenotype/data/ie_score_20um")

SPLITS = ["train", "val", "test"]

MPP = 0.25
RADIUS_UM = 20
RADIUS_PX = round(RADIUS_UM / MPP)


def read_mihc(path):
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return image.transpose(2, 0, 1)


def otsu_mask(channel):
    threshold, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return channel > threshold


def make_disk_kernel(radius_px):
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    kernel = (x * x + y * y <= radius_px * radius_px).astype(np.float32)
    return kernel / kernel.sum()


DISK_KERNEL = make_disk_kernel(RADIUS_PX)


def calculate_ie_score(cd3_mask, panck_mask):
    if cd3_mask.sum() == 0 or panck_mask.sum() == 0:
        return np.nan

    global_panck_fraction = float(panck_mask.mean())

    local_panck_map = cv2.filter2D(
        panck_mask.astype(np.float32),
        ddepth=-1,
        kernel=DISK_KERNEL,
        borderType=cv2.BORDER_REFLECT,
    )

    local_panck_fraction = max(float(local_panck_map[cd3_mask].mean()), 0.0)
    enrichment = local_panck_fraction / global_panck_fraction

    return np.log2(enrichment) if enrichment > 0 else np.nan


def process_split(split):
    mihc_dir = DATA_ROOT / split / "label"
    paths = sorted(mihc_dir.glob("*.tif")) + sorted(mihc_dir.glob("*.tiff"))

    rows = []

    for i, path in enumerate(paths, 1):
        mihc = read_mihc(path)

        cd3_mask = otsu_mask(mihc[1])
        panck_mask = otsu_mask(mihc[2])

        rows.append({
            "sample_name": path.stem,
            "split": split,
            "ie_score": calculate_ie_score(cd3_mask, panck_mask),
        })

        if i % 200 == 0:
            print(f"{split}: {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"ie_score_{split}.csv", index=False)

    print(
        f"{split}: n={len(df)} | "
        f"valid={df['ie_score'].notna().sum()} | "
        f"mean={df['ie_score'].mean():.4f} | "
        f"median={df['ie_score'].median():.4f}"
    )

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = [process_split(split) for split in SPLITS]
    all_scores = pd.concat(dfs, ignore_index=True)

    all_scores.to_csv(OUT_DIR / "ie_score_all.csv", index=False)

    np.savez(
        OUT_DIR / "ie_score_all.npz",
        sample_name=all_scores["sample_name"].to_numpy(),
        split=all_scores["split"].to_numpy(),
        ie_score=all_scores["ie_score"].to_numpy(dtype=np.float32),
    )

    print()
    print(f"Total samples: {len(all_scores)}")
    print(f"Valid IE scores: {all_scores['ie_score'].notna().sum()}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
