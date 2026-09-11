import csv
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "./result/visualization/fig2/result"
PER_PATCH_DIR = os.path.join(RESULT_DIR, "per_patch")

CUBE_CSV = os.path.join(PER_PATCH_DIR, "fig2_cube_per_patch.csv")
PIX2PIX_CSV = os.path.join(PER_PATCH_DIR, "fig2_pix2pix_resnet_per_patch.csv")
HEMIT_CSV = os.path.join(PER_PATCH_DIR, "fig2_hemit512_per_patch.csv")


def read_average(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["sample_name"]: float(row["average"]) for row in csv.DictReader(f)}


def paired(cube, other):
    names = sorted(set(cube) & set(other))
    return np.array([other[n] for n in names]), np.array([cube[n] for n in names])


def draw(ax, x, y, label):
    lo = min(-0.05, float(x.min()), float(y.min()))
    hi = min(1.0, max(0.80, float(x.max()), float(y.max())) + 0.03)

    ax.scatter(x, y, s=15, alpha=0.30, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{label} patch Pearson")
    ax.set_ylabel("CUBE patch Pearson")

    delta = y - x
    win = 100 * np.mean(delta > 0)
    ax.text(0.04, 0.96, f"CUBE > comparator: {win:.1f}%\nmedian Δr = {np.median(delta):+.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    cube = read_average(CUBE_CSV)
    pix2pix = read_average(PIX2PIX_CSV)
    hemit = read_average(HEMIT_CSV)

    x1, y1 = paired(cube, pix2pix)
    x2, y2 = paired(cube, hemit)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.1))
    draw(axes[0], x1, y1, "pix2pix ResNet")
    draw(axes[1], x2, y2, "HEMIT-512 adapted")
    fig.suptitle("Patch-wise paired performance")
    plt.tight_layout()

    fig.savefig(os.path.join(RESULT_DIR, "fig2c_paired_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULT_DIR, "fig2c_paired_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
