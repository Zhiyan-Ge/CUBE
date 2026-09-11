import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig3_common_v3 import build_real_st_method_summary, save_figure
from fig3_paths_v3 import FIG3_RESULT_DIR, FIG3_SOURCE_DIR

METRICS = [
    ("gene_pearson", "Gene Pearson"),
    ("gene_spearman", "Gene Spearman"),
    ("bin_pearson", "Bin Pearson"),
]


def main():
    data = build_real_st_method_summary()
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIG3_SOURCE_DIR / "fig3d_real_st_method_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
    y = list(range(len(data)))
    for ax, (column, label) in zip(axes, METRICS):
        ax.scatter(data[column], y, s=75)
        for i, value in enumerate(data[column]):
            ax.text(value + 0.012, i, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 0.7)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, data["method"])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Matched Visium HD held-out test | decoder-only fine-tuning with frozen H&E→UR2 encoder",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, FIG3_RESULT_DIR / "fig3d_real_st_benchmark")
    plt.close(fig)


if __name__ == "__main__":
    main()
