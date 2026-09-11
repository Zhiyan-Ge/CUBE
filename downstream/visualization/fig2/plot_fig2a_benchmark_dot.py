import csv
import os
import matplotlib.pyplot as plt

RESULT_DIR = "./result/visualization/fig2/result"
SUMMARY_CSV = os.path.join(RESULT_DIR, "fig2_benchmark_summary.csv")


def main():
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda x: float(x["mean"]))
    models = [x["model"] for x in rows]
    values = [float(x["mean"]) for x in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(values, y, s=75, zorder=3)

    for yi, value in zip(y, values):
        ax.text(value + 0.006, yi, f"{value:.3f}", va="center", fontsize=9)

    ax.set_yticks(y, models)
    for label in ax.get_yticklabels():
        if label.get_text() == "CUBE":
            label.set_fontweight("bold")

    ax.set_xlim(0.48, 0.77)
    ax.set_xlabel("Mean Pearson correlation")
    ax.set_title("H&E to mIHC benchmark")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()

    fig.savefig(os.path.join(RESULT_DIR, "fig2a_benchmark_dot.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULT_DIR, "fig2a_benchmark_dot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
