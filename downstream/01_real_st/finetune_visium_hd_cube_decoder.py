#!/usr/bin/env python3
"""Fine-tune only the CUBE UR2->ST decoder on spatially split Visium HD data."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
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
GENE_MAPPING = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_16um_paired/gene_mapping.csv"
)
OUT_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100"
)


GPU_ID = 3
GROUPS = 8
RUN_MODES = ("pretrained", "scratch")
TRAIN_PATCH_LIMIT = 100
MIN_TRAIN_FILTERED_BINS = 128
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4
MIN_DETECTED_BINS = 50
SEED = 2026
USE_AMP = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def select_training_patches(metadata: pd.DataFrame) -> list[str]:
    candidates = metadata.loc[
        (metadata["split"] == "train")
        & (metadata["filtered_bins"] >= MIN_TRAIN_FILTERED_BINS),
        "patch_id",
    ].astype(str)
    if TRAIN_PATCH_LIMIT is not None and len(candidates) > TRAIN_PATCH_LIMIT:
        candidates = candidates.sample(TRAIN_PATCH_LIMIT, random_state=SEED)
    return sorted(candidates.tolist())


def compute_target_stats(patch_ids: list[str]):
    total = np.zeros(256, dtype=np.float64)
    squared = np.zeros(256, dtype=np.float64)
    valid_bins = 0
    gene_mask = None

    for patch_id in tqdm(patch_ids, desc="Fit real-ST train statistics", dynamic_ncols=True):
        with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
            values = target["st_log1p_cp10k"][target["bin_mask"].astype(bool)].astype(np.float64)
            total += values.sum(axis=0)
            squared += np.square(values).sum(axis=0)
            valid_bins += len(values)
            if gene_mask is None:
                gene_mask = target["gene_mask"].astype(bool)

    mean = total / valid_bins
    variance = np.maximum(squared / valid_bins - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    safe_std = np.where(raw_std > 1e-6, raw_std, 1.0)
    mean[~gene_mask] = 0.0
    raw_std[~gene_mask] = 0.0
    safe_std[~gene_mask] = 1.0
    return mean.astype(np.float32), raw_std.astype(np.float32), safe_std.astype(np.float32), gene_mask, valid_bins


class UR2STDataset(Dataset):
    def __init__(self, patch_ids: list[str], mean: np.ndarray, safe_std: np.ndarray):
        self.patch_ids = patch_ids
        self.mean = mean.reshape(1, 1, 256)
        self.safe_std = safe_std.reshape(1, 1, 256)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int):
        patch_id = self.patch_ids[index]
        ur2 = np.load(UR2_FEATURE_DIR / f"{patch_id}.npy").astype(np.float32)
        with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
            expression = target["st_log1p_cp10k"].astype(np.float32)
            bin_mask = target["bin_mask"].astype(bool)
        expression = (expression - self.mean) / self.safe_std
        return torch.from_numpy(ur2), torch.from_numpy(expression), torch.from_numpy(bin_mask), patch_id


def build_loader(
    patch_ids: list[str], mean: np.ndarray, safe_std: np.ndarray,
    shuffle: bool, seed: int,
):
    dataset = UR2STDataset(patch_ids, mean, safe_std)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0, generator=generator,
    )


def load_decoder(mode: str, device: torch.device):
    sys.path.insert(0, str(CUBE_MODEL_DIR))
    from he_st.st_decoder import STDecoder

    decoder = STDecoder(GROUPS)
    checkpoint_epoch = -1
    if mode == "pretrained":
        checkpoint = torch.load(CUBE_CHECKPOINT, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
        prefix = "model3.st_decoder."
        decoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        decoder.load_state_dict(decoder_state, strict=True)
        checkpoint_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    return decoder.to(device), checkpoint_epoch


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor,
    bin_mask: torch.Tensor, gene_mask: torch.Tensor,
):
    weights = bin_mask.float().unsqueeze(-1) * gene_mask.view(1, 1, 1, -1).float()
    numerator = (torch.square(prediction.float() - target.float()) * weights).sum()
    return numerator / weights.sum(), int(weights.sum().item())


def run_epoch(
    decoder, loader, device: torch.device, gene_mask: torch.Tensor,
    optimizer=None, scaler=None, description: str = "Train",
):
    training = optimizer is not None
    decoder.train(training)
    loss_sum, element_count = 0.0, 0
    use_amp = USE_AMP and device.type == "cuda"

    for ur2, target, bin_mask, _ in tqdm(loader, desc=description, dynamic_ncols=True):
        ur2 = ur2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bin_mask = bin_mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = decoder(ur2)
                loss, valid_elements = masked_mse(prediction, target, bin_mask, gene_mask)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_sum += float(loss.item()) * valid_elements
        element_count += valid_elements
    return loss_sum / element_count


def train_decoder(
    mode: str, train_ids: list[str], val_ids: list[str],
    mean: np.ndarray, safe_std: np.ndarray, gene_mask_np: np.ndarray,
    device: torch.device,
):
    set_seed(SEED)
    decoder, checkpoint_epoch = load_decoder(mode, device)
    gene_mask = torch.from_numpy(gene_mask_np).to(device)
    train_loader = build_loader(train_ids, mean, safe_std, True, SEED)
    val_loader = build_loader(val_ids, mean, safe_std, False, SEED)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    checkpoint_path = OUT_DIR / f"best_decoder_{mode}.pt"
    history, best_loss, best_epoch, stale_epochs = [], float("inf"), 0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            decoder, train_loader, device, gene_mask, optimizer, scaler,
            f"{mode} {epoch}/{EPOCHS} train",
        )
        with torch.inference_mode():
            val_loss = run_epoch(
                decoder, val_loader, device, gene_mask,
                description=f"{mode} {epoch}/{EPOCHS} val",
            )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{mode} epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "mode": mode,
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "target_mean": mean,
                    "target_safe_std": safe_std,
                    "gene_mask": gene_mask_np,
                    "train_patch_ids": np.asarray(train_ids),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder.load_state_dict(best["decoder"], strict=True)
    pd.DataFrame(history).to_csv(OUT_DIR / f"training_history_{mode}.csv", index=False)
    return decoder.eval(), history, best_epoch, best_loss, checkpoint_epoch


def pearson_by_axis(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    n = x.shape[axis]
    sx = x.sum(axis=axis, dtype=np.float64)
    sy = y.sum(axis=axis, dtype=np.float64)
    sxx = np.square(x, dtype=np.float64).sum(axis=axis)
    syy = np.square(y, dtype=np.float64).sum(axis=axis)
    sxy = np.multiply(x, y, dtype=np.float64).sum(axis=axis)
    numerator = n * sxy - sx * sy
    denominator = np.sqrt(
        np.maximum(n * sxx - sx * sx, 0.0)
        * np.maximum(n * syy - sy * sy, 0.0)
    )
    return np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def global_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.square(x).sum() * np.square(y).sum())
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def evaluate_matrix(truth: np.ndarray, prediction: np.ndarray):
    gene_pearson = pearson_by_axis(truth, prediction, axis=0)
    gene_spearman = np.asarray([
        spearmanr(truth[:, index], prediction[:, index]).statistic
        if np.std(truth[:, index]) > 0 and np.std(prediction[:, index]) > 0 else np.nan
        for index in range(truth.shape[1])
    ])
    bin_pearson = pearson_by_axis(truth, prediction, axis=1)
    detected_bins = np.count_nonzero(truth > 0, axis=0)
    eligible = detected_bins >= MIN_DETECTED_BINS
    difference = prediction - truth
    summary = {
        "valid_bins": int(truth.shape[0]),
        "evaluated_genes": int(truth.shape[1]),
        "genes_detected_ge_50": int(eligible.sum()),
        "gene_pearson_mean_detected_ge_50": float(np.nanmean(gene_pearson[eligible])),
        "gene_pearson_median_detected_ge_50": float(np.nanmedian(gene_pearson[eligible])),
        "gene_pearson_fraction_gt_0_3": float(np.nanmean(gene_pearson[eligible] > 0.3)),
        "gene_spearman_mean_detected_ge_50": float(np.nanmean(gene_spearman[eligible])),
        "gene_spearman_median_detected_ge_50": float(np.nanmedian(gene_spearman[eligible])),
        "bin_pearson_mean": float(np.nanmean(bin_pearson)),
        "bin_pearson_median": float(np.nanmedian(bin_pearson)),
        "global_pearson": global_pearson(truth, prediction),
        "mae": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
        "negative_prediction_fraction": float(np.mean(prediction < 0)),
    }
    arrays = {
        "detected_bins": detected_bins,
        "true_mean": truth.mean(axis=0),
        "prediction_mean": prediction.mean(axis=0),
        "prediction_std": prediction.std(axis=0),
        "pearson": gene_pearson,
        "spearman": gene_spearman,
    }
    return summary, arrays


def predict_test(
    decoder, test_ids: list[str], mean: np.ndarray, safe_std: np.ndarray,
    gene_mask: np.ndarray, device: torch.device, mode: str,
):
    loader = build_loader(test_ids, mean, safe_std, False, SEED)
    predictions_z, truths, masks, names = [], [], [], []
    use_amp = USE_AMP and device.type == "cuda"

    with torch.inference_mode():
        for ur2, _, bin_mask, patch_ids in tqdm(loader, desc=f"{mode} test", dynamic_ncols=True):
            ur2 = ur2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = decoder(ur2)
            predictions_z.append(prediction.float().cpu().numpy())
            masks.append(bin_mask.numpy().astype(bool))
            names.extend(list(patch_ids))
            for patch_id in patch_ids:
                with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
                    truths.append(target["st_log1p_cp10k"].astype(np.float32))

    predictions_z = np.concatenate(predictions_z)
    truths = np.stack(truths)
    masks = np.concatenate(masks)
    predictions = predictions_z * safe_std.reshape(1, 1, 1, 256)
    predictions += mean.reshape(1, 1, 1, 256)
    truth_flat = np.concatenate([truths[i][masks[i]] for i in range(len(names))])[:, gene_mask]
    prediction_flat = np.concatenate([predictions[i][masks[i]] for i in range(len(names))])[:, gene_mask]

    np.savez_compressed(
        OUT_DIR / f"test_predictions_{mode}.npz",
        patch_ids=np.asarray(names), predictions_zscore=predictions_z,
        predictions_log1p_cp10k=predictions, truth_log1p_cp10k=truths,
        bin_masks=masks, gene_mask=gene_mask,
    )
    return truth_flat, prediction_flat


def load_pseudo_st_stats():
    with np.load(PSEUDO_ST_STATS) as stats:
        return stats["mean"].astype(np.float32), stats["safe_std"].astype(np.float32)


def save_metrics(
    mode: str, truth: np.ndarray, prediction: np.ndarray,
    gene_mapping: pd.DataFrame, gene_mask: np.ndarray,
):
    summaries = []
    metrics = gene_mapping.copy()
    variants = {"as_predicted": prediction, "clipped0": np.maximum(prediction, 0.0)}
    for variant, values in variants.items():
        summary, arrays = evaluate_matrix(truth, values)
        summary.update({"mode": mode, "variant": variant})
        summaries.append(summary)
        for field, array in arrays.items():
            full = np.full(256, np.nan, dtype=np.float64)
            full[gene_mask] = array
            metrics[f"{field}_{variant}"] = full
    metrics.to_csv(OUT_DIR / f"gene_metrics_{mode}.csv", index=False)
    return summaries


def save_loss_curves(histories: dict[str, list[dict]]):
    fig, axes = plt.subplots(1, len(histories), figsize=(7 * len(histories), 5), squeeze=False)
    for ax, (mode, history) in zip(axes.ravel(), histories.items()):
        frame = pd.DataFrame(history)
        ax.plot(frame["epoch"], frame["train_loss"], label="train")
        ax.plot(frame["epoch"], frame["val_loss"], label="val")
        ax.set_title(mode)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=250)
    plt.close(fig)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    metadata = pd.read_csv(SPATIAL_SPLIT_CSV)
    train_ids = select_training_patches(metadata)
    val_ids = sorted(metadata.loc[metadata["split"] == "val", "patch_id"].astype(str).tolist())
    test_ids = sorted(metadata.loc[metadata["split"] == "test", "patch_id"].astype(str).tolist())
    mean, raw_std, safe_std, target_gene_mask, train_valid_bins = compute_target_stats(train_ids)
    np.savez_compressed(
        OUT_DIR / "real_st_train_stats.npz", mean=mean, raw_std=raw_std,
        safe_std=safe_std, gene_mask=target_gene_mask,
        valid_bins=np.int64(train_valid_bins), patch_ids=np.asarray(train_ids),
    )

    gene_mapping = pd.read_csv(GENE_MAPPING).sort_values("model_channel").reset_index(drop=True)
    mapping_gene_mask = as_bool(gene_mapping["in_visium_hd"])
    if not np.array_equal(target_gene_mask, mapping_gene_mask):
        raise ValueError("gene_mask differs between ST targets and gene_mapping.csv")

    selected = metadata.copy()
    selected["used_for_training"] = selected["patch_id"].astype(str).isin(train_ids)
    selected.to_csv(OUT_DIR / "finetuning_patch_split.csv", index=False)

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    histories, summaries, run_reports = {}, [], []

    zero_shot_decoder, source_epoch = load_decoder("pretrained", device)
    pseudo_mean, pseudo_safe_std = load_pseudo_st_stats()
    truth, prediction = predict_test(
        zero_shot_decoder.eval(), test_ids, pseudo_mean, pseudo_safe_std,
        target_gene_mask, device, "zero_shot",
    )
    summaries.extend(save_metrics(
        "zero_shot", truth, prediction,
        gene_mapping, target_gene_mask,
    ))
    run_reports.append({
        "mode": "zero_shot",
        "source_checkpoint_epoch": source_epoch,
        "best_epoch": None,
        "best_validation_loss": None,
        "epochs_completed": 0,
    })
    del zero_shot_decoder
    for mode in RUN_MODES:
        decoder, history, best_epoch, best_loss, source_epoch = train_decoder(
            mode, train_ids, val_ids, mean, safe_std,
            target_gene_mask, device,
        )
        truth, prediction = predict_test(
            decoder, test_ids, mean, safe_std,
            target_gene_mask, device, mode,
        )
        mode_summaries = save_metrics(
            mode, truth, prediction,
            gene_mapping, target_gene_mask,
        )
        histories[mode] = history
        summaries.extend(mode_summaries)
        run_reports.append({
            "mode": mode,
            "source_checkpoint_epoch": source_epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
        })
        del decoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(summaries).to_csv(OUT_DIR / "finetuning_summary.csv", index=False)
    save_loss_curves(histories)
    report = {
        "method": "decoder-only fine-tuning on frozen CUBE UR2",
        "input_domain": "Macenko + global Lab L* normalized",
        "cube_checkpoint": str(CUBE_CHECKPOINT),
        "pseudo_st_stats_for_zero_shot_inverse": str(PSEUDO_ST_STATS),
        "spatial_split": str(SPATIAL_SPLIT_CSV),
        "train_patches": len(train_ids),
        "train_min_filtered_bins": MIN_TRAIN_FILTERED_BINS,
        "train_valid_bins": train_valid_bins,
        "validation_patches": len(val_ids),
        "test_patches": len(test_ids),
        "genes_total": 256,
        "genes_trained_and_evaluated": int(target_gene_mask.sum()),
        "missing_genes": gene_mapping.loc[~mapping_gene_mask, "gene_name"].astype(str).tolist(),
        "target": "training-split gene-wise Z-score of log1p(CP10K)",
        "loss": "masked MSE over valid bins and available genes",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "runs": run_reports,
        "results": summaries,
        "training_seconds": time.time() - start_time,
    }
    with (OUT_DIR / "finetuning_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD CUBE DECODER-ONLY FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"train/val/test patches: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"train valid bins: {train_valid_bins}")
    print(f"genes evaluated: {target_gene_mask.sum()}/256")
    for row in summaries:
        print(
            f"{row['mode']:10s} {row['variant']:12s} | "
            f"gene Pearson={row['gene_pearson_mean_detected_ge_50']:.4f} | "
            f"gene Spearman={row['gene_spearman_mean_detected_ge_50']:.4f} | "
            f"bin Pearson={row['bin_pearson_mean']:.4f}"
        )
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()
