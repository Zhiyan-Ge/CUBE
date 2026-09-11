import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


CODE_DIR = Path("./result/visualization/fig5/code")
OUTPUT_DIR = Path("./result/visualization/fig5/result/fig5a")

UMAP_CSV = Path("./result/02_UMAP/result/fig5a_fixed_coord/fig5a_fixed_coord_umap_coordinates.csv")
REPORT_JSON = Path("./result/02_UMAP/result/fig5a_fixed_coord/fig5a_fixed_coord_report.json")


SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
}


def save_figure(fig, out_prefix):
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")


def add_colorbar(fig, ax, mappable, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def plot_continuous(ax, fig, df, xcol, ycol, value_col, title, vmin, vmax, cmap="viridis"):
    sc = ax.scatter(
        df[xcol],
        df[ycol],
        c=df[value_col],
        s=6,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    add_colorbar(fig, ax, sc, title)


def plot_split(ax, df, xcol, ycol):
    for split in SPLIT_ORDER:
        sub = df[df["split"] == split]
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=6,
            c=SPLIT_COLORS[split],
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            label=f"{split} (n={len(sub)})",
        )

    ax.set_title("Data split", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best", markerscale=1.8, handletextpad=0.4, borderpad=0.2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UMAP_CSV)
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    xcol = "pca50_euclidean_UMAP1"
    ycol = "pca50_euclidean_UMAP2"
    ranges = report["color_display"]["ranges"]["pca50_euclidean"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    fig.subplots_adjust(wspace=0.28)

    plot_continuous(axes[0], fig, df, xcol, ycol, "DAPI", "DAPI", ranges["DAPI"][0], ranges["DAPI"][1], cmap="viridis")
    plot_continuous(axes[1], fig, df, xcol, ycol, "CD3", "CD3", ranges["CD3"][0], ranges["CD3"][1], cmap="viridis")
    plot_continuous(axes[2], fig, df, xcol, ycol, "panCK", "panCK", ranges["panCK"][0], ranges["panCK"][1], cmap="viridis")
    plot_continuous(axes[3], fig, df, xcol, ycol, "tissue_fraction", "Tissue fraction", ranges["tissue_fraction"][0], ranges["tissue_fraction"][1], cmap="viridis")
    plot_split(axes[4], df, xcol, ycol)

    fig.suptitle("Fig. 5A | Fixed-coordinate UMAP of fused UR representations", fontsize=14, y=1.02)
    save_figure(fig, OUTPUT_DIR / "fig5a_fixed_coord_final")
    plt.close(fig)

    source_out = OUTPUT_DIR / "fig5a_fixed_coord_visualization_source.csv"
    df.to_csv(source_out, index=False)

    print(f"Saved figure to: {OUTPUT_DIR}")
    print(f"Saved source copy to: {source_out}")


if __name__ == "__main__":
    main()