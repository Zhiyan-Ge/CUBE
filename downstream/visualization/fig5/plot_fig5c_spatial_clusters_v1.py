import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm


ASSIGN_CSV = Path("./result/03_spatial_cluster/result/fig5b_high_resolution_comparison/high_resolution_cluster_assignments_with_phenotypes.csv")
HE_DIR = Path("./result/03_spatial_cluster/result/fig5b_spatial_geometry/mosaics/all_regions")
MIHC_ROOT = Path("./result/03_spatial_cluster/result/fig5b_mihc_geometry")
OUTPUT_DIR = Path("./result/visualization/fig5/result/fig5c")

DEFAULT_SCHEME = "k8"
DEFAULT_N_REGIONS = 2
MIHC_GAMMA = 0.45
DPI = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["k7", "k8"], default=DEFAULT_SCHEME)
    parser.add_argument("--n_regions", type=int, default=DEFAULT_N_REGIONS)
    parser.add_argument("--regions", nargs="*", default=None)
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
    if image.max() > 1:
        image /= 255.0
    return np.clip(image, 0, 1)


def read_gray(path):
    image = plt.imread(path).astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    if image.max() > 1:
        image /= 255.0
    return np.clip(image, 0, 1)


def load_assignments(scheme):
    df = pd.read_csv(ASSIGN_CSV)
    cols = ["split", "sample_name", "mega_id", "patch_i", "patch_j", "DAPI", "CD3", "panCK", "tissue_fraction", scheme]
    return df[cols].copy().rename(columns={scheme: "cluster"})


def build_region_stats(df):
    stats = df.groupby(["split", "mega_id"], as_index=False).agg(
        n_patches=("sample_name", "size"),
        n_clusters_present=("cluster", "nunique"),
        dominant_fraction=("cluster", lambda x: x.value_counts(normalize=True).iloc[0]),
        tissue_fraction_mean=("tissue_fraction", "mean"),
        DAPI_mean=("DAPI", "mean"),
        CD3_mean=("CD3", "mean"),
        panCK_mean=("panCK", "mean"),
    )
    stats["selection_score"] = stats["n_clusters_present"] - stats["dominant_fraction"] + 0.1 * stats["tissue_fraction_mean"]
    return stats.sort_values(["n_clusters_present", "dominant_fraction", "tissue_fraction_mean"], ascending=[False, True, False]).reset_index(drop=True)


def select_regions(stats, user_regions, n_regions):
    if user_regions:
        selected = stats[stats["mega_id"].isin(user_regions)].copy()
        order = {region: i for i, region in enumerate(user_regions)}
        selected["manual_order"] = selected["mega_id"].map(order)
        return selected.sort_values("manual_order").drop(columns="manual_order").reset_index(drop=True)
    return stats.head(n_regions).copy().reset_index(drop=True)


def load_region_images(split, mega_id):
    stem = mega_to_stem(split, mega_id)
    he = read_rgb(HE_DIR / f"{stem}.png")
    dapi = read_gray(MIHC_ROOT / "DAPI" / f"{stem}.png")
    cd3 = read_gray(MIHC_ROOT / "CD3" / f"{stem}.png")
    panck = read_gray(MIHC_ROOT / "panCK" / f"{stem}.png")
    composite = np.power(np.clip(np.stack([panck, cd3, dapi], axis=-1), 0, 1), MIHC_GAMMA)
    return he, composite


def make_cluster_grid(region_df):
    n_rows = int(region_df["patch_i"].max()) + 1
    n_cols = int(region_df["patch_j"].max()) + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for row in region_df.itertuples():
        grid[int(row.patch_i), int(row.patch_j)] = int(row.cluster)
    return grid


def cluster_colors(n_clusters):
    cmap = plt.get_cmap("tab10")
    return [np.array(cmap(i)[:3], dtype=np.float32) for i in range(n_clusters)]


def plot_cluster_grid(ax, grid, n_clusters, target_shape):
    colors = cluster_colors(n_clusters)
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(0.5, n_clusters + 1.5), n_clusters)

    h, w = target_shape[:2]
    n_rows, n_cols = grid.shape
    x_edges = np.linspace(0, w, n_cols + 1)
    y_edges = np.linspace(0, h, n_rows + 1)

    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, interpolation="nearest", extent=(0, w, h, 0), aspect="auto")

    for x in x_edges:
        ax.plot([x, x], [0, h], color="white", linewidth=0.3)
    for y in y_edges:
        ax.plot([0, w], [y, y], color="white", linewidth=0.3)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_box_aspect(h / w)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_assignments(args.scheme)
    stats = build_region_stats(df)
    selected = select_regions(stats, args.regions, args.n_regions)

    n_clusters = int(args.scheme[1:])
    n_regions = len(selected)

    fig, axes = plt.subplots(n_regions, 3, figsize=(12.6, 3.4 * n_regions), squeeze=False)
    titles = ["H&E", "mIHC composite", "UR cluster grid"]

    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12)

    records = []

    for i, row in enumerate(selected.itertuples(index=False)):
        region_df = df[(df["split"] == row.split) & (df["mega_id"] == row.mega_id)].sort_values(["patch_i", "patch_j"])
        he, composite = load_region_images(row.split, row.mega_id)
        cluster_grid = make_cluster_grid(region_df)

        h, w = he.shape[:2]

        axes[i, 0].imshow(he)
        axes[i, 1].imshow(composite)
        plot_cluster_grid(axes[i, 2], cluster_grid, n_clusters, he.shape)

        for ax in axes[i, :2]:
            ax.set_box_aspect(h / w)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        axes[i, 0].text(-0.04, 0.5, f"Region {i + 1}", transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")

        records.append({
            "display_region": f"Region {i + 1}",
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

    colors = cluster_colors(n_clusters)
    handles = [Patch(facecolor=colors[k - 1], edgecolor="none", label=f"C{k}") for k in range(1, n_clusters + 1)]
    fig.legend(handles=handles, loc="lower center", ncol=n_clusters, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"Fig. 5C | Spatial mapping of fused UR clusters ({args.scheme.upper()})", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.09, wspace=0.08, hspace=0.12)

    prefix = OUTPUT_DIR / f"fig5c_{args.scheme}_spatial_clusters_v1"
    fig.savefig(f"{prefix}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    stats.to_csv(OUTPUT_DIR / f"fig5c_{args.scheme}_region_ranking_v1.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / f"fig5c_{args.scheme}_selected_regions_v1.csv", index=False)

    print(f"Scheme: {args.scheme}")
    print(f"Selected regions: {len(selected)}")
    print(selected[["split", "mega_id", "n_clusters_present", "dominant_fraction", "tissue_fraction_mean"]].to_string(index=False))
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()