from pathlib import Path
import json
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


PATCH_METADATA = Path("./result/01_real_st/data/HD/visium_hd_16um_paired/patch_metadata.csv")
HEMIT_TARGET_NPZ = Path("./result/01_real_st/data/ZEN47/ZEN47_macenko/hemit_train_macenko_target.npz")
HEMIT_LUMINANCE_JSON = Path("./result/01_real_st/data/ZEN47/ZEN47_macenko/luminance_trial/luminance_calibration.json")
OUT_DIR = Path("./result/01_real_st/data/HD/visium_hd_16um_macenko")

MIN_FILTERED_BINS_FOR_FIT = 128
FIT_PATCHES = 256
OD_PIXELS_PER_PATCH = 5000
L_PIXELS_PER_PATCH = 2000
NEAR_WHITE = 245
OD_THRESHOLD = 0.15
MACENKO_ALPHA = 1.0
CONCENTRATION_PERCENTILE = 99.0
RANDOM_SEED = 2026
OUTPUT_SIZES = (1024, 512, 256)
PREVIEW_PATCHES = 8


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def rgb_to_od(rgb):
    return -np.log((rgb.astype(np.float32) + 1.0) / 256.0)


def estimate_stain_matrix(od_pixels, target_matrix):
    covariance = np.cov(od_pixels, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    plane = eigenvectors[:, -2:]
    projected = od_pixels @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [MACENKO_ALPHA, 100.0 - MACENKO_ALPHA])
    vectors = np.stack([
        plane @ np.asarray([np.cos(low), np.sin(low)]),
        plane @ np.asarray([np.cos(high), np.sin(high)]),
    ], axis=1)
    vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
    for column in range(2):
        if vectors[:, column].sum() < 0:
            vectors[:, column] *= -1
    direct = np.dot(vectors[:, 0], target_matrix[:, 0]) + np.dot(vectors[:, 1], target_matrix[:, 1])
    swapped = np.dot(vectors[:, 1], target_matrix[:, 0]) + np.dot(vectors[:, 0], target_matrix[:, 1])
    return vectors if direct >= swapped else vectors[:, ::-1]


def estimate_max_concentrations(od_pixels, stain_matrix):
    concentrations = np.linalg.lstsq(stain_matrix, od_pixels.T, rcond=None)[0].T
    concentrations = np.clip(concentrations, 0, None)
    return np.percentile(concentrations, CONCENTRATION_PERCENTILE, axis=0).astype(np.float32)


def sample_od_pixels(image, count, rng):
    pixels = image.reshape(-1, 3)
    od = rgb_to_od(pixels)
    keep = np.any(pixels < NEAR_WHITE, axis=1) & (np.linalg.norm(od, axis=1) > OD_THRESHOLD)
    selected = od[keep]
    if len(selected) > count:
        selected = selected[rng.choice(len(selected), count, replace=False)]
    return selected


def normalize_macenko(image, source_matrix, source_max, target_matrix, target_max):
    shape = image.shape
    od = rgb_to_od(image).reshape(-1, 3)
    concentrations = np.linalg.lstsq(source_matrix, od.T, rcond=None)[0]
    concentrations = np.clip(concentrations, 0, None)
    concentrations *= (target_max / source_max)[:, None]
    normalized_od = target_matrix @ concentrations
    normalized = (256.0 * np.exp(-normalized_od.T) - 1.0).reshape(shape)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    background = np.all(image >= NEAR_WHITE, axis=2)
    normalized[background] = image[background]
    return normalized


def apply_luminance(image, slope, intercept):
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = np.clip(slope * lab[:, :, 0] + intercept, 0, 100)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def load_target():
    with np.load(HEMIT_TARGET_NPZ) as target:
        matrix_key = "stain_matrix" if "stain_matrix" in target.files else "target_stain_matrix"
        max_key = "max_concentrations" if "max_concentrations" in target.files else "target_max_concentrations"
        matrix = target[matrix_key].astype(np.float32)
        maximum = target[max_key].astype(np.float32)
    with HEMIT_LUMINANCE_JSON.open("r", encoding="utf-8") as f:
        luminance = json.load(f)
    return matrix, maximum, float(luminance["target_L_low"]), float(luminance["target_L_high"])


def fit_hd_source(metadata, target_matrix):
    candidates = metadata[metadata["filtered_bins"] >= MIN_FILTERED_BINS_FOR_FIT]
    selected = candidates.sample(n=min(FIT_PATCHES, len(candidates)), random_state=RANDOM_SEED).reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    pooled = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Fit HD stain source"):
        image = load_rgb(PATCH_METADATA.parent / row.patch_path)
        pixels = sample_od_pixels(image, OD_PIXELS_PER_PATCH, rng)
        if len(pixels):
            pooled.append(pixels)
    pooled = np.concatenate(pooled, axis=0)
    source_matrix = estimate_stain_matrix(pooled, target_matrix)
    source_max = estimate_max_concentrations(pooled, source_matrix)
    return source_matrix.astype(np.float32), source_max, selected


def fit_hd_luminance(selected, source_matrix, source_max, target_matrix, target_max, target_low, target_high):
    rng = np.random.default_rng(RANDOM_SEED + 1)
    values = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Fit HD luminance"):
        raw = load_rgb(PATCH_METADATA.parent / row.patch_path)
        normalized = normalize_macenko(raw, source_matrix, source_max, target_matrix, target_max)
        lab = cv2.cvtColor(normalized.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        selected_l = lab[:, :, 0][np.any(raw < NEAR_WHITE, axis=2)]
        if len(selected_l) > L_PIXELS_PER_PATCH:
            selected_l = selected_l[rng.choice(len(selected_l), L_PIXELS_PER_PATCH, replace=False)]
        values.append(selected_l)
    values = np.concatenate(values)
    source_low, source_high = np.percentile(values, [5, 95])
    slope = (target_high - target_low) / (source_high - source_low)
    intercept = target_low - slope * source_low
    return float(slope), float(intercept), float(source_low), float(source_high)


def save_outputs(metadata, source_matrix, source_max, target_matrix, target_max, slope, intercept):
    output_dirs = {size: OUT_DIR / f"patches_{size}" for size in OUTPUT_SIZES}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Normalize HD patches"):
        raw = load_rgb(PATCH_METADATA.parent / row.patch_path)
        normalized = normalize_macenko(raw, source_matrix, source_max, target_matrix, target_max)
        normalized = apply_luminance(normalized, slope, intercept)
        background = (raw.mean(axis=2) >= 235) & (np.ptp(raw, axis=2) <= 15)
        normalized[background] = 255
        record = row._asdict()
        for size in OUTPUT_SIZES:
            image = normalized if size == 1024 else cv2.resize(normalized, (size, size), interpolation=cv2.INTER_AREA)
            path = output_dirs[size] / f"{row.patch_id}.png"
            Image.fromarray(image).save(path)
            record[f"normalized_patch_{size}"] = str(path.relative_to(OUT_DIR))
        records.append(record)
    return pd.DataFrame(records)


def make_preview(metadata, output_path):
    ordered = metadata.sort_values(["filtered_fraction", "block_row", "block_col"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, PREVIEW_PATCHES).round().astype(int)
    selected = ordered.iloc[indices]
    n_rows = math.ceil(len(selected) / 2)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for index, (_, row) in enumerate(selected.iterrows()):
        row_index, pair = divmod(index, 2)
        raw_ax, normalized_ax = axes[row_index, pair * 2:pair * 2 + 2]
        raw = load_rgb(PATCH_METADATA.parent / row["patch_path"])
        normalized = load_rgb(OUT_DIR / row["normalized_patch_1024"])
        raw_ax.imshow(raw)
        normalized_ax.imshow(normalized)
        raw_ax.set_title(f"raw | {row['patch_id']} | {int(row['filtered_bins'])}/256", fontsize=8)
        normalized_ax.set_title("normalized", fontsize=8)
    fig.suptitle("Visium HD raw vs HEMIT-domain normalized patches", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(PATCH_METADATA)
    target_matrix, target_max, target_l_low, target_l_high = load_target()
    source_matrix, source_max, fit_patches = fit_hd_source(metadata, target_matrix)
    slope, intercept, source_l_low, source_l_high = fit_hd_luminance(
        fit_patches, source_matrix, source_max, target_matrix, target_max, target_l_low, target_l_high,
    )

    np.savez(
        OUT_DIR / "visium_hd_macenko_source.npz",
        stain_matrix=source_matrix,
        max_concentrations=source_max,
        fit_patch_ids=fit_patches["patch_id"].to_numpy(dtype="<U32"),
    )
    normalized_metadata = save_outputs(
        metadata, source_matrix, source_max, target_matrix, target_max, slope, intercept,
    )
    normalized_metadata.to_csv(OUT_DIR / "normalized_patch_metadata.csv", index=False)
    make_preview(normalized_metadata, OUT_DIR / "normalization_preview.png")

    report = {
        "method": "slide-level Macenko + global Lab L* calibration",
        "raw_patch_metadata": str(PATCH_METADATA),
        "hemit_target_npz": str(HEMIT_TARGET_NPZ),
        "hemit_luminance_reference": str(HEMIT_LUMINANCE_JSON),
        "patches": int(len(metadata)),
        "source_fit_patches": int(len(fit_patches)),
        "source_fit_min_filtered_bins": MIN_FILTERED_BINS_FOR_FIT,
        "source_stain_matrix": source_matrix.tolist(),
        "source_max_concentrations": source_max.tolist(),
        "target_stain_matrix": target_matrix.tolist(),
        "target_max_concentrations": target_max.tolist(),
        "target_L_q5_q95": [target_l_low, target_l_high],
        "source_macenko_L_q5_q95": [source_l_low, source_l_high],
        "luminance_slope": slope,
        "luminance_intercept": intercept,
        "output_sizes": list(OUTPUT_SIZES),
        "downsampling": "cv2.INTER_AREA",
    }
    with (OUT_DIR / "normalization_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD STAIN NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"patches: {len(metadata)}")
    print(f"source fit patches: {len(fit_patches)}")
    print(f"source max concentrations: {source_max}")
    print(f"target max concentrations: {target_max}")
    print(f"source Macenko L* q5/q95: {source_l_low:.4f} / {source_l_high:.4f}")
    print(f"target HEMIT L* q5/q95: {target_l_low:.4f} / {target_l_high:.4f}")
    print(f"L_out = {slope:.6f} * L_in + {intercept:.6f}")
    print(f"preview: {OUT_DIR / 'normalization_preview.png'}")
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()
