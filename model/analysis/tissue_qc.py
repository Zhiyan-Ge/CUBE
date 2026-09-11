import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ur import config as cfg


try:
    from scipy.stats import ks_2samp, fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Main definition of near-white background. RGB >= 245 in all three channels is near-white.
NEAR_WHITE_THRESHOLD = 245

# Report multiple thresholds without choosing a threshold based on test performance.
LOW_TISSUE_THRESHOLDS = [0.01, 0.05, 0.10, 0.25]
OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "tissue_qc")


def _normalize_key(key):
    return str(key).lower().replace("-", "").replace("_", "").replace(" ", "")


def _is_image_array(x):
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim != 3:
        return False

    shape = x.shape
    if shape[0] == 3 and shape[1] >= 64 and shape[2] >= 64:  # CHW
        return True
    if shape[2] == 3 and shape[0] >= 64 and shape[1] >= 64:  # HWC
        return True
    return False


def find_he_array(data):
    """Find the largest H&E-like image array in one pickle file."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected pickle content to be dict, got {type(data)}")

    candidates = []
    for key, value in data.items():
        if not _is_image_array(value):
            continue

        nk = _normalize_key(key)
        if "mihc" in nk:
            continue
        if "st" == nk or nk.startswith("st"):
            continue

        is_he_key = nk == "he" or nk.startswith("he") or "heimage" in nk or "imagehe" in nk
        if is_he_key:
            arr = value
            if arr.shape[0] == 3:
                h, w = arr.shape[1], arr.shape[2]
            else:
                h, w = arr.shape[0], arr.shape[1]
            candidates.append((h * w, key, arr))

    if len(candidates) == 0:
        available = {k: getattr(v, "shape", type(v)) for k, v in data.items()}
        raise KeyError("Cannot automatically find H&E image.\n" f"Available keys/shapes:\n{available}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, key, arr = candidates[0]
    return key, arr


def to_hwc_uint8(arr):
    """Convert a CHW/HWC image in [0,1] or [0,255] to an HWC float32 array in [0,255]."""
    arr = np.asarray(arr)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if np.nanmax(arr) <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr


def compute_patch_stats(img):
    """Compute exact-white, near-white, and tissue fractions for an HWC image in [0,255]."""
    exact_white = np.all(img >= 254.5, axis=2)
    near_white = np.all(img >= NEAR_WHITE_THRESHOLD, axis=2)
    exact_white_ratio = float(exact_white.mean())
    near_white_ratio = float(near_white.mean())
    tissue_fraction = 1.0 - near_white_ratio
    return {"exact_white_ratio": exact_white_ratio, "near_white_ratio": near_white_ratio,
            "tissue_fraction": tissue_fraction}


def analyze_split(split_name, pkl_dir):
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if len(files) == 0:
        raise RuntimeError(f"No pkl files found for {split_name}: {pkl_dir}")

    print("=" * 78)
    print(f"Analyzing {split_name} | N={len(files)}")
    print(f"PKL dir: {pkl_dir}")

    rows = []
    detected_key = None
    for i, path in enumerate(files, start=1):
        with open(path, "rb") as f:
            data = pickle.load(f)

        he_key, he = find_he_array(data)
        if detected_key is None:
            detected_key = he_key
            print(f"Detected H&E key: {he_key}")
            print(f"Detected H&E shape: {he.shape}")

        img = to_hwc_uint8(he)
        stats = compute_patch_stats(img)
        sample_name = os.path.splitext(os.path.basename(path))[0]
        rows.append({"split": split_name, "sample_name": sample_name, "he_key": he_key, **stats})

        if i % 500 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    return pd.DataFrame(rows)


def describe_split(df, split):
    x = df["tissue_fraction"].values
    w = df["near_white_ratio"].values
    row = {
        "split": split, "N": len(df),
        "tissue_mean": np.mean(x), "tissue_median": np.median(x),
        "tissue_p01": np.percentile(x, 1), "tissue_p05": np.percentile(x, 5),
        "tissue_p10": np.percentile(x, 10), "tissue_p25": np.percentile(x, 25),
        "near_white_mean": np.mean(w), "near_white_median": np.median(w),
        "near_white_p90": np.percentile(w, 90), "near_white_p95": np.percentile(w, 95),
        "near_white_p99": np.percentile(w, 99)
    }
    for thr in LOW_TISSUE_THRESHOLDS:
        label = f"tissue_lt_{int(thr * 100)}pct"
        count = int(np.sum(x < thr))
        ratio = count / len(x)
        row[f"{label}_count"] = count
        row[f"{label}_ratio"] = ratio
    return row


def print_split_summary(df, split):
    tissue = df["tissue_fraction"].values
    print()
    print("-" * 78)
    print(f"{split} tissue statistics")
    print("-" * 78)
    print(f"Tissue fraction | mean={np.mean(tissue):.4f} | median={np.median(tissue):.4f}")
    print(f"percentiles     | P1={np.percentile(tissue, 1):.4f} | "
          f"P5={np.percentile(tissue, 5):.4f} | P10={np.percentile(tissue, 10):.4f} | "
          f"P25={np.percentile(tissue, 25):.4f}")
    for thr in LOW_TISSUE_THRESHOLDS:
        n = int(np.sum(tissue < thr))
        pct = 100.0 * n / len(tissue)
        print(f"Tissue < {thr * 100:>4.0f}% | {n:4d}/{len(tissue):4d} ({pct:6.2f}%)")


def compare_val_test(val_df, test_df):
    print()
    print("=" * 78)
    print("VAL vs TEST")
    print("=" * 78)

    val_x = val_df["tissue_fraction"].values
    test_x = test_df["tissue_fraction"].values
    print(f"Median tissue fraction | val={np.median(val_x):.4f} | test={np.median(test_x):.4f}")

    if HAS_SCIPY:
        ks = ks_2samp(val_x, test_x)
        print(f"KS test | statistic={ks.statistic:.4f} | p={ks.pvalue:.4g}")
    else:
        print("scipy not installed -> skipping KS/Fisher tests")

    print()
    for thr in LOW_TISSUE_THRESHOLDS:
        val_low = int(np.sum(val_x < thr))
        test_low = int(np.sum(test_x < thr))
        val_pct = 100 * val_low / len(val_x)
        test_pct = 100 * test_low / len(test_x)

        if val_pct > 0:
            enrichment = test_pct / val_pct
            enrich_str = f"{enrichment:.2f}x"
        else:
            enrich_str = "inf"

        line = (f"Tissue < {thr * 100:>4.0f}% | VAL {val_low:4d}/{len(val_x)} ({val_pct:6.2f}%) | "
                f"TEST {test_low:4d}/{len(test_x)} ({test_pct:6.2f}%) | Test/Val={enrich_str}")
        if HAS_SCIPY:
            table = np.array([[val_low, len(val_x) - val_low],
                              [test_low, len(test_x) - test_low]])
            _, p = fisher_exact(table)
            line += f" | Fisher p={p:.4g}"
        print(line)


def save_histogram(all_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for split in ["train", "val", "test"]:
        x = all_df.loc[all_df["split"] == split, "tissue_fraction"].values
        plt.hist(x, bins=50, alpha=0.45, density=True, label=split)
    plt.xlabel("H&E tissue fraction")
    plt.ylabel("Density")
    plt.title("Tissue fraction distribution")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tissue_fraction_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def save_low_tissue_histogram(all_df):
    """Plot the 0-30% tissue region."""
    plt.figure(figsize=(8, 5))
    for split in ["train", "val", "test"]:
        x = all_df.loc[all_df["split"] == split, "tissue_fraction"].values
        x = x[x <= 0.30]
        plt.hist(x, bins=30, alpha=0.45, density=False, label=split)
    plt.xlabel("H&E tissue fraction")
    plt.ylabel("Patch count")
    plt.title("Low-tissue patches (0-30%)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "low_tissue_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_dirs = {"train": cfg.TRAIN_PKL_DIR, "val": cfg.VAL_PKL_DIR, "test": cfg.TEST_PKL_DIR}
    dfs = {}
    for split, pkl_dir in split_dirs.items():
        df = analyze_split(split, pkl_dir)
        dfs[split] = df
        print_split_summary(df, split)

    all_df = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    patch_csv = os.path.join(OUTPUT_DIR, "tissue_fraction_per_patch.csv")
    all_df.to_csv(patch_csv, index=False)

    summary = pd.DataFrame([describe_split(dfs["train"], "train"),
                            describe_split(dfs["val"], "val"),
                            describe_split(dfs["test"], "test")])
    summary_csv = os.path.join(OUTPUT_DIR, "tissue_fraction_summary.csv")
    summary.to_csv(summary_csv, index=False)

    compare_val_test(dfs["val"], dfs["test"])
    lowest = all_df.sort_values("tissue_fraction", ascending=True).head(100)
    low_csv = os.path.join(OUTPUT_DIR, "lowest_tissue_100.csv")
    lowest.to_csv(low_csv, index=False)

    save_histogram(all_df)
    save_low_tissue_histogram(all_df)

    print()
    print("=" * 78)
    print("Finished.")
    print(f"Per-patch: {patch_csv}")
    print(f"Summary:   {summary_csv}")
    print(f"Lowest100: {low_csv}")
    print(f"Figures:   {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()