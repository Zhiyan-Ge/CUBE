import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig3_common_v3 import METHOD_LABELS, load_gene_metric_tables, save_figure
from fig3_paths_v3 import (
    FIG3_CACHE_DIR,
    FIG3F_GENES,
    FIG3_RESULT_DIR,
    FIG3_SOURCE_DIR,
    GENE_MAPPING,
    PATCH_METADATA,
    ST_TARGET_DIR,
)

METHODS_ALL = ["deepspot", "zero_shot", "pretrained", "scratch", "reset_head"]
METHODS_COMPACT = ["zero_shot", "scratch", "reset_head"]
DISPLAY_QUANTILE = 0.99


def load_geometry(patch_ids):
    raw, all_rows, all_cols = {}, [], []
    for patch_id in patch_ids:
        with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
            mask = target["bin_mask"].astype(bool)
            rows = target["array_row"][mask].astype(int)
            cols = target["array_col"][mask].astype(int)
        raw[patch_id] = (mask, rows, cols)
        all_rows.append(rows)
        all_cols.append(cols)

    row_max = int(np.concatenate(all_rows).max())
    row_min = int(np.concatenate(all_rows).min())
    col_min = int(np.concatenate(all_cols).min())
    col_max = int(np.concatenate(all_cols).max())
    shape = (row_max - row_min + 1, col_max - col_min + 1)
    geometry = {
        patch_id: (mask, row_max - rows, cols - col_min)
        for patch_id, (mask, rows, cols) in raw.items()
    }
    return geometry, shape


def empty_maps(shape):
    return {gene: np.full(shape, np.nan, dtype=np.float32) for gene in FIG3F_GENES}


def truth_maps(patch_ids, geometry, shape, channels):
    maps = empty_maps(shape)
    for patch_id in patch_ids:
        mask, rows, cols = geometry[patch_id]
        with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
            values = target["st_log1p_cp10k"]
            for gene in FIG3F_GENES:
                maps[gene][rows, cols] = values[..., channels[gene]][mask]
    return maps


def prediction_maps(method, patch_ids, geometry, shape):
    with np.load(FIG3_CACHE_DIR / f"fig3f_whole_slide_{method}.npz") as data:
        cache_ids = data["patch_ids"].astype(str)
        cache_genes = data["gene_names"].astype(str).tolist()
        prediction = data["predictions_log1p_cp10k"].astype(np.float32)
    index = {patch_id: i for i, patch_id in enumerate(cache_ids)}
    gene_index = {gene: i for i, gene in enumerate(cache_genes)}
    maps = empty_maps(shape)
    for patch_id in patch_ids:
        mask, rows, cols = geometry[patch_id]
        values = prediction[index[patch_id]]
        for gene in FIG3F_GENES:
            maps[gene][rows, cols] = values[..., gene_index[gene]][mask]
    return maps


def test_r(tables, method, gene):
    row = tables[method].set_index("gene_name").loc[gene]
    return float(row["pearson_as_predicted"])


def build_source_table(tables, methods):
    rows = []
    for gene in FIG3F_GENES:
        row = {"gene_name": gene}
        for method in methods:
            row[f"heldout_test_pearson_{method}"] = test_r(tables, method, gene)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(mode):
    methods = METHODS_ALL if mode == "all" else METHODS_COMPACT
    metadata = pd.read_csv(PATCH_METADATA).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    patch_ids = metadata["patch_id"].astype(str).tolist()
    mapping = pd.read_csv(GENE_MAPPING).set_index("gene_name")
    channels = {gene: int(mapping.loc[gene, "model_channel"]) for gene in FIG3F_GENES}
    geometry, shape = load_geometry(patch_ids)
    truth = truth_maps(patch_ids, geometry, shape, channels)
    predictions = {method: prediction_maps(method, patch_ids, geometry, shape) for method in methods}
    tables = load_gene_metric_tables()

    source = build_source_table(tables, methods)
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIG3_SOURCE_DIR / f"fig3f_selected_genes_{mode}.csv", index=False)

    columns = ["Real ST"] + [METHOD_LABELS[m] for m in methods]
    fig, axes = plt.subplots(
        len(FIG3F_GENES), len(columns),
        figsize=(2.7 * len(columns), 2.55 * len(FIG3F_GENES)),
        constrained_layout=True, squeeze=False,
    )

    for row, gene in enumerate(FIG3F_GENES):
        truth_values = truth[gene][np.isfinite(truth[gene])]
        vmax = max(float(np.quantile(truth_values, DISPLAY_QUANTILE)), 1e-6)
        image = axes[row, 0].imshow(truth[gene], vmin=0, vmax=vmax, interpolation="nearest")
        axes[row, 0].set_title(f"{gene}\nReal ST")
        axes[row, 0].axis("off")

        for col, method in enumerate(methods, start=1):
            axes[row, col].imshow(predictions[method][gene], vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(f"{METHOD_LABELS[method]}\ntest r={test_r(tables, method, gene):.3f}")
            axes[row, col].axis("off")
        fig.colorbar(image, ax=axes[row].tolist(), fraction=0.012, pad=0.005, label="log1p(CP10K)")

    fig.suptitle(
        "Whole-slide qualitative maps | display floor = 0; r values use only the held-out test stripe",
        fontsize=13,
    )
    save_figure(fig, FIG3_RESULT_DIR / f"fig3f_whole_slide_{mode}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "compact"], default="compact")
    args = parser.parse_args()
    plot(args.mode)


if __name__ == "__main__":
    main()
