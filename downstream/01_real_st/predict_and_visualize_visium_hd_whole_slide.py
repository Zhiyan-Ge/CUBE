#!/usr/bin/env python3
"""Run all CUBE decoders on every Visium HD patch and draw whole-slide gene maps."""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Edit each path independently when files move.
CUBE_MODEL_DIR = Path(
    "./model"
)
CUBE_CHECKPOINT = Path(
    "./"
    "models/test_models_3/test13/he_st/best.pt"
)
PSEUDO_ST_STATS = Path(
    "./data/train_data/final_data/train/st_stats.npz"
)
PATCH_METADATA = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_16um_paired/patch_metadata.csv"
)
SPATIAL_SPLIT_CSV = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/prepared/spatial_split.csv"
)
UR2_FEATURE_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/prepared/ur2_features"
)
ST_TARGET_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_16um_paired/st_targets"
)

ZERO_SHOT_METRICS = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/gene_metrics_zero_shot.csv"
)
PRETRAINED_CHECKPOINT = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/best_decoder_pretrained.pt"
)
PRETRAINED_METRICS = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/gene_metrics_pretrained.csv"
)
SCRATCH_CHECKPOINT = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/best_decoder_scratch.pt"
)
SCRATCH_METRICS = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/gene_metrics_scratch.csv"
)
RESET_HEAD_CHECKPOINT = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/best_decoder_reset_head.pt"
)
RESET_HEAD_METRICS = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/gene_metrics_reset_head.csv"
)

OUT_DIR = Path(
    "./result/01_real_st/"
    "result/HD/visium_hd_whole_slide_prediction"
)


GPU_ID = 3
GROUPS = 8
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 2026
USE_AMP = True
SKIP_EXISTING = True
SAVE_DTYPE = np.float16

TOP_GENES_PER_METHOD = 2
MIN_DETECTED_BINS = 50
VARIANT = "clipped0"
MANUAL_GENES = []
CMAP = "magma"
TRUTH_COLOR_QUANTILE = 0.99
DPI = 250


METHODS = {
    "zero_shot": {
        "label": "CUBE zero-shot",
        "checkpoint": None,
        "metrics": ZERO_SHOT_METRICS,
    },
    "pretrained": {
        "label": "CUBE pretrained FT",
        "checkpoint": PRETRAINED_CHECKPOINT,
        "metrics": PRETRAINED_METRICS,
    },
    "scratch": {
        "label": "CUBE scratch FT",
        "checkpoint": SCRATCH_CHECKPOINT,
        "metrics": SCRATCH_METRICS,
    },
    "reset_head": {
        "label": "CUBE reset-head FT",
        "checkpoint": RESET_HEAD_CHECKPOINT,
        "metrics": RESET_HEAD_METRICS,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def load_metrics():
    tables = {}
    for method, config in METHODS.items():
        table = pd.read_csv(config["metrics"]).sort_values("model_channel").reset_index(drop=True)
        table["in_visium_hd"] = as_bool(table["in_visium_hd"])
        tables[method] = table
    return tables


def select_genes(tables):
    pearson_col = f"pearson_{VARIANT}"
    detected_col = f"detected_bins_{VARIANT}"
    selected, selected_by = [], {}
    for method, table in tables.items():
        eligible = table.loc[
            table["in_visium_hd"] & (table[detected_col] >= MIN_DETECTED_BINS)
        ].sort_values(pearson_col, ascending=False)
        for gene in eligible.head(TOP_GENES_PER_METHOD)["gene_name"].astype(str):
            if gene not in selected:
                selected.append(gene)
            selected_by.setdefault(gene, []).append(method)
    if MANUAL_GENES:
        selected = list(dict.fromkeys(MANUAL_GENES))
        selected_by = {gene: ["manual"] for gene in selected}
    return selected, selected_by


class FeatureDataset(Dataset):
    def __init__(self, patch_ids):
        self.patch_ids = patch_ids

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]
        feature = np.load(UR2_FEATURE_DIR / f"{patch_id}.npy").astype(np.float32)
        return torch.from_numpy(feature), patch_id


def build_loader(patch_ids):
    return DataLoader(
        FeatureDataset(patch_ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def load_original_decoder(device):
    sys.path.insert(0, str(CUBE_MODEL_DIR))
    from he_st.st_decoder import STDecoder

    checkpoint = torch.load(CUBE_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    prefix = "model3.st_decoder."
    decoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    decoder = STDecoder(GROUPS)
    decoder.load_state_dict(decoder_state, strict=True)
    with np.load(PSEUDO_ST_STATS) as stats:
        mean = stats["mean"].astype(np.float32)
        safe_std = stats["safe_std"].astype(np.float32)
    return decoder.to(device).eval(), mean, safe_std


def load_finetuned_decoder(path, device):
    sys.path.insert(0, str(CUBE_MODEL_DIR))
    from he_st.st_decoder import STDecoder

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    decoder = STDecoder(GROUPS)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    safe_std = np.asarray(checkpoint["target_safe_std"], dtype=np.float32)
    return decoder.to(device).eval(), mean, safe_std


def prediction_file(method):
    return OUT_DIR / f"whole_slide_selected_predictions_{method}.npz"


def existing_predictions(method, patch_ids, genes):
    path = prediction_file(method)
    if not SKIP_EXISTING or not path.exists():
        return None
    with np.load(path) as data:
        if list(data["gene_names"].astype(str)) != genes:
            return None
        if list(data["patch_ids"].astype(str)) != patch_ids:
            return None
        return data["predictions_log1p_cp10k"].astype(np.float32)


def run_method(method, config, patch_ids, genes, channels, device):
    cached = existing_predictions(method, patch_ids, genes)
    if cached is not None:
        print(f"Loaded existing whole-slide predictions: {method}")
        return cached

    if method == "zero_shot":
        decoder, mean, safe_std = load_original_decoder(device)
    else:
        decoder, mean, safe_std = load_finetuned_decoder(config["checkpoint"], device)

    selected = []
    channel_indices = np.asarray([channels[gene] for gene in genes], dtype=np.int64)
    use_amp = USE_AMP and device.type == "cuda"
    with torch.inference_mode():
        for features, _ in tqdm(build_loader(patch_ids), desc=f"Whole slide {method}", dynamic_ncols=True):
            features = features.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction_z = decoder(features)
            prediction = prediction_z.float().cpu().numpy()
            prediction = prediction * safe_std.reshape(1, 1, 1, 256)
            prediction += mean.reshape(1, 1, 1, 256)
            if VARIANT == "clipped0":
                prediction = np.maximum(prediction, 0.0)
            selected.append(prediction[..., channel_indices])

    predictions = np.concatenate(selected).astype(np.float32)
    np.savez_compressed(
        prediction_file(method), patch_ids=np.asarray(patch_ids),
        gene_names=np.asarray(genes), model_channels=channel_indices,
        predictions_log1p_cp10k=predictions.astype(SAVE_DTYPE),
    )
    del decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


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


def empty_maps(genes, shape):
    return {gene: np.full(shape, np.nan, dtype=np.float32) for gene in genes}


def build_truth_maps(patch_ids, geometry, shape, genes, channels):
    maps = empty_maps(genes, shape)
    for patch_id in patch_ids:
        mask, rows, cols = geometry[patch_id]
        with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
            expression = target["st_log1p_cp10k"]
            for gene in genes:
                maps[gene][rows, cols] = expression[..., channels[gene]][mask]
    return maps


def build_prediction_maps(predictions, patch_ids, geometry, shape, genes):
    maps = empty_maps(genes, shape)
    for index, patch_id in enumerate(patch_ids):
        mask, rows, cols = geometry[patch_id]
        for gene_index, gene in enumerate(genes):
            maps[gene][rows, cols] = predictions[index, ..., gene_index][mask]
    return maps


def metric_value(tables, method, gene):
    table = tables[method].set_index("gene_name")
    return float(table.loc[gene, f"pearson_{VARIANT}"])


def gene_vmax(truth_map):
    values = truth_map[np.isfinite(truth_map)]
    return max(float(np.quantile(values, TRUTH_COLOR_QUANTILE)), 1e-6)


def show_map(ax, values, title, vmax, cmap):
    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    return image


def save_selected_gene_table(genes, selected_by, tables):
    reference = tables["zero_shot"].set_index("gene_name")
    records = []
    for gene in genes:
        record = {
            "gene_name": gene,
            "model_channel": int(reference.loc[gene, "model_channel"]),
            "selected_by": ";".join(
                METHODS[m]["label"] if m in METHODS else m for m in selected_by[gene]
            ),
        }
        for method in METHODS:
            record[f"heldout_test_pearson_{method}"] = metric_value(tables, method, gene)
        records.append(record)
    pd.DataFrame(records).to_csv(OUT_DIR / "whole_slide_selected_genes.csv", index=False)


def plot_split_context(metadata):
    colors = {"train": "#4daf4a", "val": "#ffb000", "test": "#377eb8", "unused": "#bdbdbd"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for split in ["train", "val", "test", "unused"]:
        part = metadata.loc[metadata["split"] == split]
        ax.scatter(
            part["block_col"], -part["block_row"], s=40,
            c=colors[split], label=f"{split} (n={len(part)})",
            edgecolors="black", linewidths=0.2,
        )
    ax.set_xlabel("block_col")
    ax.set_ylabel("spatial block row")
    ax.set_title("Whole-slide visualization context")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "whole_slide_split_context.png", dpi=DPI)
    plt.close(fig)


def plot_zero_shot_overview(genes, truth_maps, method_maps, tables, cmap):
    fig, axes = plt.subplots(
        len(genes), 2, figsize=(8, 2.8 * len(genes)), constrained_layout=True, squeeze=False
    )
    for row, gene in enumerate(genes):
        vmax = gene_vmax(truth_maps[gene])
        image = show_map(axes[row, 0], truth_maps[gene], f"{gene}\nReal ST", vmax, cmap)
        r = metric_value(tables, "zero_shot", gene)
        show_map(
            axes[row, 1], method_maps["zero_shot"][gene],
            f"CUBE zero-shot\nheld-out test r={r:.3f}", vmax, cmap,
        )
        fig.colorbar(image, ax=axes[row].tolist(), fraction=0.022, pad=0.01, label="log1p(CP10K)")
    fig.suptitle("Whole-slide zero-shot prediction", fontsize=16)
    fig.savefig(OUT_DIR / "whole_slide_zero_shot_overview.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods(genes, truth_maps, method_maps, tables, cmap):
    columns = ["Real ST"] + [config["label"] for config in METHODS.values()]
    fig, axes = plt.subplots(
        len(genes), len(columns), figsize=(3.0 * len(columns), 2.75 * len(genes)),
        constrained_layout=True, squeeze=False,
    )
    for row, gene in enumerate(genes):
        vmax = gene_vmax(truth_maps[gene])
        image = show_map(axes[row, 0], truth_maps[gene], f"{gene}\nReal ST", vmax, cmap)
        for col, method in enumerate(METHODS, start=1):
            r = metric_value(tables, method, gene)
            show_map(
                axes[row, col], method_maps[method][gene],
                f"{METHODS[method]['label']}\ntest r={r:.3f}", vmax, cmap,
            )
        fig.colorbar(image, ax=axes[row].tolist(), fraction=0.012, pad=0.006, label="log1p(CP10K)")
    fig.suptitle(
        "Whole-slide qualitative maps | r values use the held-out test stripe only", fontsize=16
    )
    fig.savefig(OUT_DIR / "whole_slide_all_cube_methods.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_individual_genes(genes, truth_maps, method_maps, tables, cmap):
    gene_dir = OUT_DIR / "gene_maps"
    gene_dir.mkdir(exist_ok=True)
    for gene in genes:
        fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(16, 7), constrained_layout=True)
        vmax = gene_vmax(truth_maps[gene])
        image = show_map(axes[0], truth_maps[gene], "Real ST", vmax, cmap)
        for col, method in enumerate(METHODS, start=1):
            r = metric_value(tables, method, gene)
            show_map(
                axes[col], method_maps[method][gene],
                f"{METHODS[method]['label']}\ntest r={r:.3f}", vmax, cmap,
            )
        fig.colorbar(image, ax=axes.tolist(), fraction=0.016, pad=0.008, label="log1p(CP10K)")
        fig.suptitle(f"{gene} | whole-slide qualitative visualization", fontsize=16)
        fig.savefig(gene_dir / f"{gene}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(PATCH_METADATA).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    split = pd.read_csv(SPATIAL_SPLIT_CSV)[["patch_id", "split"]]
    metadata = metadata.drop(columns=["split"], errors="ignore").merge(split, on="patch_id", how="left")
    patch_ids = metadata["patch_id"].astype(str).tolist()

    tables = load_metrics()
    genes, selected_by = select_genes(tables)
    channels = {
        str(row.gene_name): int(row.model_channel)
        for row in tables["zero_shot"].itertuples()
    }
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    predictions = {
        method: run_method(method, config, patch_ids, genes, channels, device)
        for method, config in METHODS.items()
    }

    geometry, shape = load_geometry(patch_ids)
    truth_maps = build_truth_maps(patch_ids, geometry, shape, genes, channels)
    method_maps = {
        method: build_prediction_maps(values, patch_ids, geometry, shape, genes)
        for method, values in predictions.items()
    }
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("white")

    save_selected_gene_table(genes, selected_by, tables)
    plot_split_context(metadata)
    plot_zero_shot_overview(genes, truth_maps, method_maps, tables, cmap)
    plot_all_methods(genes, truth_maps, method_maps, tables, cmap)
    plot_individual_genes(genes, truth_maps, method_maps, tables, cmap)

    print("=" * 80)
    print("VISIUM HD WHOLE-SLIDE CUBE VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"patches predicted: {len(patch_ids)}")
    print(f"selected genes: {', '.join(genes)}")
    print("metrics shown in figures come from the held-out test stripe only")
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()
