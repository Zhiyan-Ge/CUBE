import csv
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "./result/visualization/fig2/result"
SUMMARY_CSV = os.path.join(RESULT_DIR, "fig2_benchmark_summary.csv")
MARKERS = ["DAPI", "CD3", "panCK"]


def main():
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models = [x["model"] for x in rows]
    values = np.array([[float(x[m]) for m in MARKERS] for x in rows])

    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    im = ax.imshow(values, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(MARKERS)), MARKERS)
    ax.set_yticks(range(len(models)), models)
    for label in ax.get_yticklabels():
        if label.get_text() == "CUBE":
            label.set_fontweight("bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < 0.45 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    ax.set_title("Marker-resolved performance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    plt.tight_layout()

    fig.savefig(os.path.join(RESULT_DIR, "fig2b_marker_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULT_DIR, "fig2b_marker_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
