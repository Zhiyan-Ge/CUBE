import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ASSIGN_CSV = Path("./result/03_spatial_cluster/result/fig5b_high_resolution_comparison/high_resolution_cluster_assignments_with_phenotypes.csv")
UMAP_CSV = Path("./result/02_UMAP/result/fig5a_fixed_coord/fig5a_fixed_coord_umap_coordinates.csv")
OUTPUT_DIR = Path("./result/visualization/fig5/result/fig5b")

DEFAULT_SCHEME = "k8"
DPI = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["k7", "k8"], default=DEFAULT_SCHEME)
    return parser.parse_args()


def cluster_colors(n_clusters):
    cmap = plt.get_cmap("tab10")
    return [np.array(cmap(i)[:3], dtype=np.float32) for i in range(n_clusters)]


def load_data(scheme):
    assign = pd.read_csv(ASSIGN_CSV)[["split", "sample_name", scheme]].rename(columns={scheme: "cluster"})
    umap = pd.read_csv(UMAP_CSV)[["split", "sample_name", "pca50_euclidean_UMAP1", "pca50_euclidean_UMAP2"]]
    return assign.merge(umap, on=["split", "sample_name"], how="inner")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.scheme)
    n_clusters = int(args.scheme[1:])
    colors = cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    for cluster in range(1, n_clusters + 1):
        sub = df[df["cluster"] == cluster]
        ax.scatter(
            sub["pca50_euclidean_UMAP1"],
            sub["pca50_euclidean_UMAP2"],
            s=8,
            color=colors[cluster - 1],
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )

    x_min, x_max = df["pca50_euclidean_UMAP1"].min(), df["pca50_euclidean_UMAP1"].max()
    y_min, y_max = df["pca50_euclidean_UMAP2"].min(), df["pca50_euclidean_UMAP2"].max()
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(f"Fig. 5B | Fused UR clusters ({args.scheme.upper()})", fontsize=14)

    handles = [Patch(facecolor=colors[k - 1], edgecolor="none", label=f"C{k}") for k in range(1, n_clusters + 1)]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=n_clusters)

    fig.tight_layout()
    prefix = OUTPUT_DIR / f"fig5b_{args.scheme}_cluster_umap_v1"
    fig.savefig(f"{prefix}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUTPUT_DIR / f"fig5b_{args.scheme}_cluster_umap_source_v1.csv", index=False)

    print(f"Scheme: {args.scheme}")
    print(f"N patches: {len(df)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()