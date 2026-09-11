import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from ur.dataset import URDataset
from ur.model import URModel
from ur import config as cfg
from loss.concept_loss import concept_loss


def set_seed(seed):
    """Random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_samples(dataset, n, seed):
    """Select a random subset of the dataset with n samples, using the given seed for reproducibility."""
    if n is None or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, indices)


def build_dataset(pkl_dir, ur1_dir, ur2_dir, n_samples, seed):
    """Build a URDataset and optionally select a random subset of samples."""
    dataset = URDataset(pkl_dir, ur1_dir, ur2_dir)
    return select_samples(dataset, n_samples, seed)


def build_loader(dataset, shuffle):
    """Build a DataLoader for the given dataset with specified shuffle option."""
    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )


def run_epoch(model, loader, device, optimizer=None, scaler=None, desc="Train"):
    """Run one epoch of training or evaluation."""
    training = optimizer is not None
    model.train(training)
    loss_sum, sample_count = 0.0, 0

    pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        ur1 = batch["ur1"].to(device, non_blocking=True)
        ur2 = batch["ur2"].to(device, non_blocking=True)
        coord = batch["coord"].to(device, non_blocking=True)
        target = batch["concept"].to(device, non_blocking=True)
        batch_size = ur1.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                pred = model(ur1, ur2, coord)
                loss = concept_loss(pred, target)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return loss_sum / sample_count


def save_checkpoint(model, optimizer, epoch, path):
    """Save the model and optimizer states."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch": epoch,
        "model": target.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)


def save_curve(history):
    """Save the training and validation loss curves."""
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train"], label="Train")

    if cfg.USE_VALIDATION:
        plt.plot(epochs, history["val"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("UR Concept Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "concept_loss.png"), dpi=200)
    plt.close()


def save_report(history, train_size, val_size, best_epoch, best_loss, elapsed):
    """Save the training configuration, final metrics, and per-epoch losses."""
    report = {
        "train_samples": train_size,
        "validation_enabled": cfg.USE_VALIDATION,
        "val_samples": val_size,
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "weight_decay": cfg.WEIGHT_DECAY,
        "gpu_ids": cfg.GPU_IDS,
        "training_seconds": elapsed,
        "model": {
            "dim": cfg.DIM,
            "heads": cfg.HEADS,
            "ffn_dim": cfg.FFN_DIM,
            "dropout": cfg.DROPOUT
        },
        "loss": "concept MSE",
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_metric": "val_loss" if cfg.USE_VALIDATION else "train_loss",
        "final_train": history["train"][-1],
        "final_val": history["val"][-1] if cfg.USE_VALIDATION else None,
        "history": history
    }

    with open(os.path.join(cfg.OUTPUT_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    set_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # Data
    train_set = build_dataset(
        cfg.TRAIN_PKL_DIR, cfg.TRAIN_UR1_DIR, cfg.TRAIN_UR2_DIR,
        cfg.TRAIN_SAMPLES, cfg.SEED
    )
    val_set = build_dataset(
        cfg.VAL_PKL_DIR, cfg.VAL_UR1_DIR, cfg.VAL_UR2_DIR,
        cfg.VAL_SAMPLES, cfg.SEED + 1
    ) if cfg.USE_VALIDATION else None

    train_loader = build_loader(train_set, shuffle=True)
    val_loader = build_loader(val_set, shuffle=False) if val_set else None

    # Model
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}")
    model = URModel(cfg.DIM, cfg.HEADS, cfg.FFN_DIM, cfg.DROPOUT).to(device)
    if len(cfg.GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=cfg.GPU_IDS)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    history = {"train": [], "val": []}
    best_loss, best_epoch = float("inf"), 0

    print("=" * 70)
    print(f"UR training | train={len(train_set)} | validation={cfg.USE_VALIDATION}")
    print(f"GPU={cfg.GPU_IDS} | batch={cfg.BATCH_SIZE} | epochs={cfg.EPOCHS}")
    print("=" * 70)

    # Training
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer, scaler,
            desc=f"Epoch {epoch}/{cfg.EPOCHS} Train"
        )
        history["train"].append(train_loss)

        if cfg.USE_VALIDATION:
            val_loss = run_epoch(
                model, val_loader, device,
                desc=f"Epoch {epoch}/{cfg.EPOCHS} Val"
            )
            history["val"].append(val_loss)
            metric = val_loss
            print(f"Epoch {epoch} | train={train_loss:.6f} | val={val_loss:.6f}")
        else:
            metric = train_loss
            print(f"Epoch {epoch} | train={train_loss:.6f}")

        if metric < best_loss:
            best_loss, best_epoch = metric, epoch
            save_checkpoint(model, optimizer, epoch, os.path.join(cfg.OUTPUT_DIR, "best.pt"))

        if epoch % cfg.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pt")
            )

    # Results
    save_checkpoint(model, optimizer, cfg.EPOCHS, os.path.join(cfg.OUTPUT_DIR, "last.pt"))
    save_curve(history)
    save_report(
        history, len(train_set), len(val_set) if val_set else 0,
        best_epoch, best_loss, time.time() - start_time
    )

    print("=" * 70)
    print(f"Training finished | best epoch={best_epoch} | best loss={best_loss:.6f}")
    print(f"Results: {cfg.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()