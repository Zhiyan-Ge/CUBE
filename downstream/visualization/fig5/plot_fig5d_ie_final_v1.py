from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


PROBE_DIR = Path("./result/04_interface_phenotype/result/ie_mlp_probe")
BASELINE_DIR = Path("./result/04_interface_phenotype/result/ie_control_baselines")
OUTPUT_DIR = Path("./result/visualization/fig5/result/fig5d")

FUSION_PRED_CSV = PROBE_DIR / "fusion" / "test_predictions.csv"
COMPARISON_CSV = BASELINE_DIR / "ie_probe_combined_comparison.csv"

REPRESENTATION_ORDER = ["UR1", "UR2", "Fusion", "Concat", "Concepts"]
METRICS = ["pearson", "spearman", "r2"]
METRIC_LABELS = ["Pearson", "Spearman", "R²"]
DPI = 300


def save_figure(fig, prefix):
    fig.savefig(f"{prefix}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")


def load_data():
    fusion = pd.read_csv(FUSION_PRED_CSV)
    comparison = pd.read_csv(COMPARISON_CSV)
    comparison["representation"] = pd.Categorical(comparison["representation"], categories=REPRESENTATION_ORDER, ordered=True)
    comparison = comparison.sort_values("representation").reset_index(drop=True)
    return fusion, comparison


def plot_fusion_scatter(ax, df):
    y_true = df["ie_score"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    pearson = pearsonr(y_true, y_pred).statistic
    spearman = spearmanr(y_true, y_pred).statistic
    r2 = r2_score(y_true, y_pred)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05

    ax.scatter(y_true, y_pred, s=26, alpha=0.65, linewidths=0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured IE score")
    ax.set_ylabel("Predicted IE score")
    ax.set_title("Fusion: held-out test prediction", fontsize=12, pad=8)

    text = f"Pearson = {pearson:.3f}\nSpearman = {spearman:.3f}\nR² = {r2:.3f}\nN = {len(df)}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=10)

    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)

    return {"pearson": pearson, "spearman": spearman, "r2": r2, "N": len(df)}


def plot_probe_comparison(ax, df):
    x = np.arange(len(df))
    width = 0.22

    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[metric], width, label=label)

        for bar, value in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    ax.set_xticks(x, df["representation"])
    ax.set_ylim(0, 0.93)
    ax.set_ylabel("Held-out test performance")
    ax.set_title("Representation comparison", fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        borderaxespad=0,
        columnspacing=1.8,
        handletextpad=0.7,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fusion, comparison = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.0, 1.35]})

    scatter_metrics = plot_fusion_scatter(axes[0], fusion)
    plot_probe_comparison(axes[1], comparison)

    fig.suptitle("Fig. 5D | Decoding immune–epithelial spatial enrichment from frozen representations", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.20, wspace=0.22)

    save_figure(fig, OUTPUT_DIR / "fig5d_ie_final_v2")
    plt.close(fig)

    fusion.to_csv(OUTPUT_DIR / "fig5d_fusion_scatter_source_v2.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "fig5d_probe_comparison_source_v2.csv", index=False)

    pd.DataFrame([{
        "representation": "Fusion",
        "N": scatter_metrics["N"],
        "pearson": scatter_metrics["pearson"],
        "spearman": scatter_metrics["spearman"],
        "r2": scatter_metrics["r2"],
    }]).to_csv(OUTPUT_DIR / "fig5d_fusion_scatter_metrics_v2.csv", index=False)

    print()
    print("=" * 80)
    print("FIG5D FUSION HELD-OUT TEST")
    print("=" * 80)
    print(f"N        = {scatter_metrics['N']}")
    print(f"Pearson  = {scatter_metrics['pearson']:.4f}")
    print(f"Spearman = {scatter_metrics['spearman']:.4f}")
    print(f"R2       = {scatter_metrics['r2']:.4f}")

    print()
    print("=" * 80)
    print("FIG5D REPRESENTATION COMPARISON")
    print("=" * 80)
    print(comparison[["representation", "pearson", "spearman", "r2"]].to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()