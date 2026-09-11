#!/usr/bin/env python3
"""Plot Fig. 5B region-level spatial cluster panels for CUBE fused-UR."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ASSIGN_CSV = Path("./result/03_spatial_cluster/result/fig5b_high_resolution_comparison/high_resolution_cluster_assignments_with_phenotypes.csv")
HE_DIR = Path("./result/03_spatial_cluster/result/fig5b_spatial_geometry/mosaics/all_regions")
MIHC_ROOT = Path("./result/03_spatial_cluster/result/fig5b_mihc_geometry")
OUT_ROOT = Path("./result/03_spatial_cluster/result/fig5b_final_panels")

PATCH_SIZE = 512
STRIDE = 256
DEFAULT_SCHEME = "k7"
DEFAULT_N_REGIONS = 4
FIGURE_DPI = 300
MIHC_GAMMA = 0.35


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["k7", "k8"], default=DEFAULT_SCHEME)
    parser.add_argument("--n_regions", type=int, default=DEFAULT_N_REGIONS)
    parser.add_argument("--regions", nargs="*", default=None, help='Optional mega_id list, e.g. "[16266,48808]" "[13215,61369]"')
    return parser.parse_args()


def mega_to_stem(split, mega_id):
    x, y = re.findall(r"\d+", mega_id)
    return f"{split}_{x}_{y}"


def read_rgb(path):
    image = plt.imread(path).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.clip(image, 0, 1)


def read_gray(path):
    image = plt.imread(path).astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    return np.clip(image, 0, 1)


def load_assignments(scheme):
    df = pd.read_csv(ASSIGN_CSV)
    df = df[["split", "sample_name", "mega_id", "patch_i", "patch_j", "DAPI", "CD3", "panCK", "tissue_fraction", scheme]].copy()
    df = df.rename(columns={scheme: "cluster"})
    return df


def build_region_stats(df):
    stats = df.groupby(["split", "mega_id"], as_index=False).agg(
        n_patches=("sample_name", "size"),
        n_clusters_present=("cluster", "nunique"),
        dominant_fraction=("cluster", lambda s: s.value_counts(normalize=True).iloc[0]),
        tissue_fraction_mean=("tissue_fraction", "mean"),
        DAPI_mean=("DAPI", "mean"),
        CD3_mean=("CD3", "mean"),
        panCK_mean=("panCK", "mean"),
    )
    stats["selection_score"] = stats["n_clusters_present"] - stats["dominant_fraction"] + 0.1 * stats["tissue_fraction_mean"]
    stats = stats.sort_values(
        ["n_clusters_present", "dominant_fraction", "tissue_fraction_mean", "selection_score"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)
    return stats


def select_regions(stats, user_regions, n_regions):
    if user_regions:
        wanted = set(user_regions)
        selected = stats[stats["mega_id"].isin(wanted)].copy()
        return selected.reset_index(drop=True)
    selected = []
    used = set()
    for row in stats.itertuples():
        if row.mega_id in used:
            continue
        selected.append(row)
        used.add(row.mega_id)
        if len(selected) == n_regions:
            break
    return pd.DataFrame(selected)


def load_region_images(split, mega_id):
    stem = mega_to_stem(split, mega_id)
    he = read_rgb(HE_DIR / f"{stem}.png")
    dapi = read_gray(MIHC_ROOT / "DAPI" / f"{stem}.png")
    cd3 = read_gray(MIHC_ROOT / "CD3" / f"{stem}.png")
    panck = read_gray(MIHC_ROOT / "panCK" / f"{stem}.png")
    composite = np.power(np.clip(np.stack([panck, cd3, dapi], axis=-1), 0, 1), MIHC_GAMMA)
    return he, dapi, cd3, panck, composite


def make_cluster_vote_map(region_df, n_clusters):
    n_rows = int(region_df["patch_i"].max()) + 1
    n_cols = int(region_df["patch_j"].max()) + 1
    height = PATCH_SIZE + (n_rows - 1) * STRIDE
    width = PATCH_SIZE + (n_cols - 1) * STRIDE
    votes = np.zeros((n_clusters, height, width), dtype=np.float32)
    cover = np.zeros((height, width), dtype=np.float32)

    for row in region_df.itertuples():
        y = int(row.patch_i) * STRIDE
        x = int(row.patch_j) * STRIDE
        k = int(row.cluster) - 1
        votes[k, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1
        cover[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1

    label_map = votes.argmax(axis=0) + 1
    confidence = np.divide(votes.max(axis=0), cover, out=np.zeros_like(cover), where=cover > 0)
    label_map[cover == 0] = 0
    return label_map, confidence


def cluster_colors(n_clusters):
    cmap = plt.get_cmap("tab10")
    colors = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
    colors += [np.array(cmap(i)[:3], dtype=np.float32) for i in range(n_clusters)]
    return np.stack(colors, axis=0)


def label_to_rgb(label_map, n_clusters):
    colors = cluster_colors(n_clusters)
    return colors[label_map]


def make_overlay(he, label_map, confidence, n_clusters):
    cluster_rgb = label_to_rgb(label_map, n_clusters)
    alpha = np.where(label_map > 0, 0.18 + 0.37 * confidence, 0.0).astype(np.float32)
    overlay = he * (1 - alpha[..., None]) + cluster_rgb * alpha[..., None]
    return np.clip(overlay, 0, 1)


def plot_main_figure(selected_regions, df, scheme, out_dir):
    n_clusters = int(scheme[1:])
    n_rows = len(selected_regions)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["H&E", "mIHC composite", f"{scheme.upper()} cluster map", "H&E + cluster overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold")

    region_records = []
    for i, row in enumerate(selected_regions.itertuples(index=False)):
        region_df = df[(df["split"] == row.split) & (df["mega_id"] == row.mega_id)].copy().sort_values(["patch_i", "patch_j"])
        he, dapi, cd3, panck, composite = load_region_images(row.split, row.mega_id)
        label_map, confidence = make_cluster_vote_map(region_df, n_clusters)
        cluster_map = label_to_rgb(label_map, n_clusters)
        overlay = make_overlay(he, label_map, confidence, n_clusters)

        images = [he, composite, cluster_map, overlay]
        for j, image in enumerate(images):
            axes[i, j].imshow(image)
            axes[i, j].axis("off")

        title = f"{row.split} | {row.mega_id}\nclusters={row.n_clusters_present} | dominant={row.dominant_fraction:.3f}"
        axes[i, 0].text(-0.02, 0.5, title, transform=axes[i, 0].transAxes, va="center", ha="right", fontsize=10)

        plt.imsave(out_dir / "individual_panels" / f"{row.split}_{row.mega_id.replace('[', '').replace(']', '').replace(',', '_')}_he.png", he)
        plt.imsave(out_dir / "individual_panels" / f"{row.split}_{row.mega_id.replace('[', '').replace(']', '').replace(',', '_')}_mihc.png", composite)
        plt.imsave(out_dir / "individual_panels" / f"{row.split}_{row.mega_id.replace('[', '').replace(']', '').replace(',', '_')}_cluster.png", cluster_map)
        plt.imsave(out_dir / "individual_panels" / f"{row.split}_{row.mega_id.replace('[', '').replace(']', '').replace(',', '_')}_overlay.png", overlay)

        region_records.append({
            "split": row.split,
            "mega_id": row.mega_id,
            "n_patches": row.n_patches,
            "n_clusters_present": row.n_clusters_present,
            "dominant_fraction": row.dominant_fraction,
            "tissue_fraction_mean": row.tissue_fraction_mean,
            "DAPI_mean": row.DAPI_mean,
            "CD3_mean": row.CD3_mean,
            "panCK_mean": row.panCK_mean,
        })

    legend_handles = [Patch(facecolor=cluster_colors(n_clusters)[k], edgecolor="none", label=f"C{k}") for k in range(1, n_clusters + 1)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(n_clusters, 8), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"CUBE Fig. 5B region-level spatial visualization ({scheme.upper()})", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / f"{scheme}_region_panels.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{scheme}_region_panels.pdf", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(region_records).to_csv(out_dir / f"{scheme}_selected_regions.csv", index=False)


def plot_cluster_summary(df, scheme, out_dir):
    summary = df.groupby("cluster", as_index=False).agg(
        n=("sample_name", "size"),
        fraction=("sample_name", lambda s: len(s) / len(df)),
        DAPI_median=("DAPI", "median"),
        CD3_median=("CD3", "median"),
        panCK_median=("panCK", "median"),
        tissue_fraction_median=("tissue_fraction", "median"),
    )
    summary = summary.sort_values("cluster").reset_index(drop=True)
    summary.to_csv(out_dir / f"{scheme}_cluster_summary.csv", index=False)

    heat = summary[["DAPI_median", "CD3_median", "panCK_median", "tissue_fraction_median"]].to_numpy(dtype=np.float32)
    cols = ["DAPI", "CD3", "panCK", "tissue"]
    rows = [f"C{c}" for c in summary["cluster"].tolist()]

    fig, ax = plt.subplots(figsize=(6, 0.7 * len(rows) + 2))
    im = ax.imshow(heat, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(f"{scheme.upper()} cluster phenotype summary", fontsize=14, fontweight="bold")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if heat[i, j] > 0.45 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Median value")
    fig.tight_layout()
    fig.savefig(out_dir / f"{scheme}_cluster_summary_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{scheme}_cluster_summary_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = OUT_ROOT / args.scheme
    (out_dir / "individual_panels").mkdir(parents=True, exist_ok=True)

    df = load_assignments(args.scheme)
    stats = build_region_stats(df)
    stats.to_csv(out_dir / f"{args.scheme}_region_ranking.csv", index=False)

    selected = select_regions(stats, args.regions, args.n_regions)
    selected = selected.reset_index(drop=True)
    plot_main_figure(selected, df, args.scheme, out_dir)
    plot_cluster_summary(df, args.scheme, out_dir)

    print(f"scheme: {args.scheme}")
    print(f"selected regions: {len(selected)}")
    print(selected[["split", "mega_id", "n_clusters_present", "dominant_fraction"]].to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
