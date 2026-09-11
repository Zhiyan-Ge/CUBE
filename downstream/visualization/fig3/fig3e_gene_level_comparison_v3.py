import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig3_common_v3 import METHOD_LABELS, load_gene_metric_tables, save_figure
from fig3_paths_v3 import FIG3_RESULT_DIR, FIG3_SOURCE_DIR

PAIRS = [
    ("deepspot", "zero_shot", "DeepSpot → CUBE zero-shot"),
    ("zero_shot", "reset_head", "CUBE zero-shot → reset-head FT"),
    ("scratch", "reset_head", "Scratch FT → reset-head FT"),
]


def paired_frame(left, right, tables):
    a = tables[left][["gene_name", "in_visium_hd", "detected_bins_as_predicted", "pearson_as_predicted"]].copy()
    b = tables[right][["gene_name", "pearson_as_predicted"]].copy()
    a = a.rename(columns={"pearson_as_predicted": "left_r"})
    b = b.rename(columns={"pearson_as_predicted": "right_r"})
    merged = a.merge(b, on="gene_name", how="inner")
    return merged.loc[
        merged["in_visium_hd"] & (merged["detected_bins_as_predicted"] >= 50)
        & np.isfinite(merged["left_r"]) & np.isfinite(merged["right_r"])
    ].reset_index(drop=True)


def main():
    tables = load_gene_metric_tables()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    summaries = []

    for ax, (left, right, title) in zip(axes, PAIRS):
        data = paired_frame(left, right, tables)
        delta = data["right_r"] - data["left_r"]
        win = float((delta > 0).mean())
        median_delta = float(np.median(delta))
        method_corr = float(np.corrcoef(data["left_r"], data["right_r"])[0, 1])
        summaries.append({
            "left": METHOD_LABELS[left],
            "right": METHOD_LABELS[right],
            "n_genes": len(data),
            "right_win_fraction": win,
            "median_delta_r": median_delta,
            "cross_gene_method_correlation": method_corr,
        })

        low = min(float(data[["left_r", "right_r"]].min().min()), -0.1)
        high = max(float(data[["left_r", "right_r"]].max().max()), 0.5)
        pad = 0.05 * (high - low)
        low, high = low - pad, high + pad
        ax.scatter(data["left_r"], data["right_r"], s=17, alpha=0.7)
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(METHOD_LABELS[left])
        ax.set_ylabel(METHOD_LABELS[right])
        ax.set_title(title, fontsize=10)
        ax.text(
            0.04, 0.96,
            f"N={len(data)}\nright > left: {win:.1%}\nmedian Δr={median_delta:+.3f}\nmethod r={method_corr:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        )

    summary = pd.DataFrame(summaries)
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIG3_SOURCE_DIR / "fig3e_gene_pairwise_summary.csv", index=False)
    fig.suptitle("Per-gene performance on the matched Visium HD test stripe", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG3_RESULT_DIR / "fig3e_gene_level_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
