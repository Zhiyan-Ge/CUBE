from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr


PROBE_DIR = Path("./result/05_real_st_program/result/program_probes")
OUTPUT_DIR = Path("./result/visualization/fig5/result/fig5e")

PRED_CSV = PROBE_DIR / "fig5d_probe_predictions.csv"
METRIC_CSV = PROBE_DIR / "fig5d_probe_metrics.csv"

REPRESENTATIONS = ["UR1", "UR2", "Fusion"]
DPI = 300


def save_figure(fig, prefix):
    fig.savefig(f"{prefix}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")


def make_spatial_matrix(df, value_col):
    n_rows = int(df["block_row"].max()) + 1
    n_cols = int(df["block_col"].max()) + 1
    matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for row in df.itertuples():
        matrix[int(row.block_row), int(row.block_col)] = getattr(row, value_col)
    return matrix


def plot_spatial_map(ax, matrix, title, vmin, vmax, show_ylabel=True):
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", origin="upper", aspect="equal")
    ax.set_title(title, fontsize=12, pad=7)
    ax.set_xlabel("Spatial block column")
    ax.set_ylabel("Spatial block row" if show_ylabel else "")
    ax.spines[["top", "right"]].set_visible(False)

    test_start = 20.5
    test_width = 5
    rect = Rectangle((test_start, -0.5), test_width, matrix.shape[0], fill=False, edgecolor="black", linewidth=1.0, linestyle="--")
    ax.add_patch(rect)

    return im


def plot_test_scatter(ax, df):
    test = df[df["split"] == "test"].copy()
    y_true = test["ecm_true"].to_numpy()
    y_pred = test["ecm_fusion_pred"].to_numpy()

    pearson = pearsonr(y_true, y_pred).statistic
    spearman = spearmanr(y_true, y_pred).statistic

    ax.scatter(y_true, y_pred, s=30, alpha=0.70, linewidths=0)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1)

    ax.set_xlabel("Measured ECM score")
    ax.set_ylabel("Predicted ECM score", labelpad=6)
    ax.set_title("Fusion: held-out test", fontsize=12, pad=7)

    ax.text(
        0.04, 0.96,
        f"Pearson = {pearson:.3f}\nSpearman = {spearman:.3f}\nN = {len(test)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="0.3", boxstyle="round,pad=0.25"),
    )

    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)

    return pearson, spearman, len(test)


def plot_pearson_comparison(ax, metrics):
    sub = metrics[(metrics["target"] == "ECM") & (metrics["representation"].isin(REPRESENTATIONS))].copy()
    order = {name: i for i, name in enumerate(REPRESENTATIONS)}
    sub["order"] = sub["representation"].map(order)
    sub = sub.sort_values("order")

    x = np.arange(len(sub))
    bars = ax.bar(x, sub["test_pearson"].to_numpy(), width=0.62)

    for bar, value in zip(bars, sub["test_pearson"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, sub["representation"])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Test Pearson r")
    ax.set_title("Representation comparison", fontsize=12, pad=7)
    ax.grid(axis="y", alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)

    return sub


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PRED_CSV)
    metrics = pd.read_csv(METRIC_CSV)

    true_map = make_spatial_matrix(predictions, "ecm_true")
    fusion_map = make_spatial_matrix(predictions, "ecm_fusion_pred")

    all_values = np.concatenate([
        predictions["ecm_true"].dropna().to_numpy(),
        predictions["ecm_fusion_pred"].dropna().to_numpy(),
    ])
    vmin = np.nanpercentile(all_values, 2)
    vmax = np.nanpercentile(all_values, 98)

    fig = plt.figure(figsize=(13.8, 4.7))
    outer = fig.add_gridspec(1, 3, width_ratios=[2.25, 1.20, 0.85], wspace=0.28)

    map_grid = outer[0, 0].subgridspec(2, 2, height_ratios=[1.0, 0.055], hspace=0.18, wspace=0.18)
    ax_true = fig.add_subplot(map_grid[0, 0])
    ax_pred = fig.add_subplot(map_grid[0, 1])
    cax = fig.add_subplot(map_grid[1, :])

    ax_scatter = fig.add_subplot(outer[0, 1])
    ax_bar = fig.add_subplot(outer[0, 2])

    im = plot_spatial_map(ax_true, true_map, "Real-ST ECM score", vmin, vmax, show_ylabel=True)
    plot_spatial_map(ax_pred, fusion_map, "Fusion-predicted ECM", vmin, vmax, show_ylabel=False)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("ECM program score", fontsize=9, labelpad=3)
    cbar.ax.tick_params(labelsize=8)

    pearson, spearman, n_test = plot_test_scatter(ax_scatter, predictions)
    comparison = plot_pearson_comparison(ax_bar, metrics)

    fig.suptitle(
        "Fig. 5E | External real-ST validation of ECM-associated transcriptomic information",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.85, bottom=0.15)

    save_figure(fig, OUTPUT_DIR / "fig5e_ecm_final_v3")
    plt.close(fig)

    predictions[
        ["patch_id", "block_row", "block_col", "split", "ecm_true", "ecm_ur1_pred", "ecm_ur2_pred", "ecm_fusion_pred"]
    ].to_csv(OUTPUT_DIR / "fig5e_ecm_spatial_source_v3.csv", index=False)

    predictions[predictions["split"] == "test"][
        ["patch_id", "block_row", "block_col", "ecm_true", "ecm_ur1_pred", "ecm_ur2_pred", "ecm_fusion_pred"]
    ].to_csv(OUTPUT_DIR / "fig5e_ecm_test_predictions_v3.csv", index=False)

    comparison.to_csv(OUTPUT_DIR / "fig5e_ecm_pearson_comparison_v3.csv", index=False)

    print()
    print("=" * 80)
    print("FIG5E ECM HELD-OUT TEST")
    print("=" * 80)
    print(f"Fusion Pearson  = {pearson:.4f}")
    print(f"Fusion Spearman = {spearman:.4f}")
    print(f"N test          = {n_test}")
    print()
    print(comparison[["representation", "test_pearson", "test_spearman", "test_r2"]].to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()