#!/usr/bin/env python3
"""Fine-tune a pretrained CUBE ST decoder after resetting only its output head."""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import finetune_visium_hd_cube_decoder as core


MODE = "reset_head"


def load_reset_head_decoder(device: torch.device):
    # Copy the output-head initialization used by the same-seed scratch control.
    core.set_seed(core.SEED)
    scratch, _ = core.load_decoder("scratch", device)
    head_state = {
        key: value.detach().cpu().clone()
        for key, value in scratch.output.state_dict().items()
    }
    del scratch

    decoder, checkpoint_epoch = core.load_decoder("pretrained", device)
    decoder.output.load_state_dict(head_state, strict=True)
    return decoder, checkpoint_epoch


def train_reset_head(
    train_ids: list[str], val_ids: list[str],
    mean: np.ndarray, safe_std: np.ndarray,
    gene_mask_np: np.ndarray, device: torch.device,
):
    decoder, checkpoint_epoch = load_reset_head_decoder(device)
    gene_mask = torch.from_numpy(gene_mask_np).to(device)
    train_loader = core.build_loader(train_ids, mean, safe_std, True, core.SEED)
    val_loader = core.build_loader(val_ids, mean, safe_std, False, core.SEED)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=core.LEARNING_RATE,
        weight_decay=core.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=core.USE_AMP and device.type == "cuda"
    )
    checkpoint_path = core.OUT_DIR / "best_decoder_reset_head.pt"
    history, best_loss, best_epoch, stale_epochs = [], float("inf"), 0, 0

    for epoch in range(1, core.EPOCHS + 1):
        train_loss = core.run_epoch(
            decoder, train_loader, device, gene_mask,
            optimizer, scaler,
            f"reset-head {epoch}/{core.EPOCHS} train",
        )
        with torch.inference_mode():
            val_loss = core.run_epoch(
                decoder, val_loader, device, gene_mask,
                description=f"reset-head {epoch}/{core.EPOCHS} val",
            )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(
            f"reset-head epoch {epoch:03d} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_loss - core.MIN_DELTA:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save({
                "epoch": epoch,
                "mode": MODE,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "target_mean": mean,
                "target_safe_std": safe_std,
                "gene_mask": gene_mask_np,
                "train_patch_ids": np.asarray(train_ids),
                "reset_module": "output",
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= core.EARLY_STOPPING_PATIENCE:
                break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder.load_state_dict(best["decoder"], strict=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        core.OUT_DIR / "training_history_reset_head.csv", index=False
    )
    return decoder.eval(), history_frame, best_epoch, best_loss, checkpoint_epoch


def save_curve(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE")
    ax.set_title("CUBE reset-head decoder fine-tuning")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(core.OUT_DIR / "training_curve_reset_head.png", dpi=250)
    plt.close(fig)


def main():
    core.set_seed(core.SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    start_time = time.time()

    split = pd.read_csv(core.OUT_DIR / "finetuning_patch_split.csv")
    val_ids = sorted(
        split.loc[split["split"] == "val", "patch_id"].astype(str).tolist()
    )
    test_ids = sorted(
        split.loc[split["split"] == "test", "patch_id"].astype(str).tolist()
    )
    with np.load(core.OUT_DIR / "real_st_train_stats.npz") as stats:
        mean = stats["mean"].astype(np.float32)
        safe_std = stats["safe_std"].astype(np.float32)
        gene_mask = stats["gene_mask"].astype(bool)
        train_ids = stats["patch_ids"].astype(str).tolist()
        train_valid_bins = int(stats["valid_bins"])

    gene_mapping = pd.read_csv(core.GENE_MAPPING).sort_values(
        "model_channel"
    ).reset_index(drop=True)
    device = torch.device(
        f"cuda:{core.GPU_ID}" if torch.cuda.is_available() else "cpu"
    )
    decoder, history, best_epoch, best_loss, checkpoint_epoch = train_reset_head(
        train_ids, val_ids, mean, safe_std, gene_mask, device
    )
    truth, prediction = core.predict_test(
        decoder, test_ids, mean, safe_std,
        gene_mask, device, MODE,
    )
    summaries = core.save_metrics(
        MODE, truth, prediction, gene_mapping, gene_mask
    )
    pd.DataFrame(summaries).to_csv(
        core.OUT_DIR / "reset_head_summary.csv", index=False
    )
    save_curve(history)

    report = {
        "method": "pretrained CUBE decoder trunk + reset output head",
        "trainable_parameters": "entire ST decoder",
        "reset_module": "STDecoder.output Linear(256, 256)",
        "head_initialization": "same-seed scratch decoder output head",
        "source_checkpoint": str(core.CUBE_CHECKPOINT),
        "source_checkpoint_epoch": checkpoint_epoch,
        "train_patches": len(train_ids),
        "train_valid_bins": train_valid_bins,
        "validation_patches": len(val_ids),
        "test_patches": len(test_ids),
        "genes_evaluated": int(gene_mask.sum()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "training_seconds": time.time() - start_time,
        "results": summaries,
    }
    with (
        core.OUT_DIR / "reset_head_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD CUBE RESET-HEAD FINE-TUNING COMPLETE")
    print("=" * 80)
    print(
        f"train/val/test patches: "
        f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
    )
    print(f"best epoch: {best_epoch} | best val loss: {best_loss:.6f}")
    for row in summaries:
        print(
            f"{row['variant']:12s} | "
            f"gene Pearson={row['gene_pearson_mean_detected_ge_50']:.4f} | "
            f"gene Spearman={row['gene_spearman_mean_detected_ge_50']:.4f} | "
            f"bin Pearson={row['bin_pearson_mean']:.4f}"
        )
    print(f"results: {core.OUT_DIR}")


if __name__ == "__main__":
    main()
