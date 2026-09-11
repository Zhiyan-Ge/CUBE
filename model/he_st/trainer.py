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

from he_st.dataset import HESTDataset
from he_st.model import HESTBranch
from he_st import config as cfg
from loss.recon_loss import he_recon_loss, st_recon_loss
from loss.alignment_loss import ur_alignment_loss
from loss.model_loss import model_loss, branch_loss


LOSS_NAMES = ["total", "he_he", "he_st", "st_he", "st_st", "ur2"]


def set_seed(seed):
    """Random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_samples(dataset, n, seed):
    """Select n samples from the dataset randomly, using a fixed seed for reproducibility."""
    if n is None or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, indices)


def build_dataset(pkl_dir, n_samples, seed):
    """Eastablish the HESTDataset and select a subset of samples."""
    dataset = HESTDataset(pkl_dir, cfg.ST_STATS_PATH)
    return select_samples(dataset, n_samples, seed)


def build_loader(dataset, shuffle):
    """Build a DataLoader for the given dataset with specified batch size and number of workers."""
    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )


def compute_losses(outputs, he, st):
    """Compute the individual losses and the total loss based on the model outputs and ground truth."""
    he_he = he_recon_loss(outputs["he_to_he"], he, cfg.ALPHA1)
    he_st = st_recon_loss(outputs["he_to_st"], st)
    st_he = he_recon_loss(outputs["st_to_he"], he, cfg.ALPHA1)
    st_st = st_recon_loss(outputs["st_to_st"], st)
    ur2 = ur_alignment_loss(outputs["ur2_he"], outputs["ur2_st"])

    m3 = model_loss(he_he, he_st, cfg.BETA3)
    m4 = model_loss(st_st, st_he, cfg.BETA4)
    total = branch_loss(m3, m4, ur2, cfg.GAMMA2)

    return {
        "total": total, "he_he": he_he, "he_st": he_st,
        "st_he": st_he, "st_st": st_st, "ur2": ur2
    }


def run_epoch(model, loader, device, optimizer=None, scaler=None, desc="Train"):
    """Run a single epoch of training or validation."""
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0

    pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        he = batch["he"].to(device, non_blocking=True)
        st = batch["st"].to(device, non_blocking=True)
        batch_size = he.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                outputs = model(he, st)
                losses = compute_losses(outputs, he, st)

            if training:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

        for name in LOSS_NAMES:
            sums[name] += losses[name].item() * batch_size
        sample_count += batch_size

        pbar.set_postfix(
            total=f"{losses['total'].item():.3f}",
            HE_ST=f"{losses['he_st'].item():.3f}",
            ST_ST=f"{losses['st_st'].item():.3f}",
            UR2=f"{losses['ur2'].item():.3f}"
        )

    return {name: value / sample_count for name, value in sums.items()}


def save_checkpoint(model, optimizer, epoch, path):
    """Save the model and optimizer state to a checkpoint file."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch": epoch,
        "model": target.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)


def load_checkpoint(model, path, device):
    """Load the model parameters for final UR2 extraction."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(torch.load(path, map_location=device)["model"])


def save_ur2(model, dataset, device, split):
    """Save the UR2 features for the given dataset split (train or validation) to files."""
    save_dir = os.path.join(cfg.OUTPUT_DIR, "ur2_he", split)
    os.makedirs(save_dir, exist_ok=True)
    loader = build_loader(dataset, shuffle=False)
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Save {split} UR2", unit="batch", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            st = batch["st"].to(device, non_blocking=True)
            ur2 = model(he, st)["ur2_he"].cpu().numpy()

            for name, feature in zip(batch["sample_name"], ur2):
                np.save(os.path.join(save_dir, f"{name}.npy"), feature)


def save_curves(history):
    """Save the training and validation loss curves to PNG files."""
    epochs = np.arange(1, len(history["train"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [x["total"] for x in history["train"]], label="Train")
    if cfg.USE_VALIDATION:
        plt.plot(epochs, [x["total"] for x in history["val"]], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HE-ST Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "total_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name in LOSS_NAMES[1:]:
        plt.plot(epochs, [x[name] for x in history["train"]], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HE-ST Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "loss_components.png"), dpi=200)
    plt.close()


def save_report(history, train_size, val_size, best_epoch, best_loss, elapsed):
    """Save a JSON report summarizing the training process, including dataset sizes, hyperparameters, best epoch, and loss history."""
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
        "st_normalization": "gene-wise Z-score",
        "st_loss": "MSE",
        "loss_weights": {
            "alpha1": cfg.ALPHA1, "beta3": cfg.BETA3,
            "beta4": cfg.BETA4, "gamma2": cfg.GAMMA2
        },
        "best_epoch": best_epoch,
        "best_he_st": best_loss,
        "best_loss": best_loss,
        "best_metric": "val_he_st" if cfg.USE_VALIDATION else "train_he_st",
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
    train_set = build_dataset(cfg.TRAIN_DIR, cfg.TRAIN_SAMPLES, cfg.SEED)
    val_set = build_dataset(cfg.VAL_DIR, cfg.VAL_SAMPLES, cfg.SEED + 1) if cfg.USE_VALIDATION else None
    train_loader = build_loader(train_set, shuffle=True)
    val_loader = build_loader(val_set, shuffle=False) if val_set else None

    # Model
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}")
    model = HESTBranch(cfg.GROUPS).to(device)
    if len(cfg.GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=cfg.GPU_IDS)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    history = {"train": [], "val": []}
    best_loss, best_epoch = float("inf"), 0

    print("=" * 70)
    print(f"HE-ST training | train={len(train_set)} | validation={cfg.USE_VALIDATION}")
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
            metric = val_loss["he_st"]
            print(f"Epoch {epoch} | train_total={train_loss['total']:.4f} | val_total={val_loss['total']:.4f} | val_he_st={metric:.4f}")
        else:
            metric = train_loss["he_st"]
            print(f"Epoch {epoch} | train_total={train_loss['total']:.4f} | train_he_st={metric:.4f}")

        if metric < best_loss:
            best_loss, best_epoch = metric, epoch
            save_checkpoint(model, optimizer, epoch, os.path.join(cfg.OUTPUT_DIR, "best.pt"))

        if epoch % cfg.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pt")
            )

    # Training results
    save_checkpoint(model, optimizer, cfg.EPOCHS, os.path.join(cfg.OUTPUT_DIR, "last.pt"))
    save_curves(history)
    save_report(
        history, len(train_set), len(val_set) if val_set else 0,
        best_epoch, best_loss, time.time() - start_time
    )

    # Save UR2 features if configured
    if cfg.SAVE_UR2:
        load_checkpoint(model, os.path.join(cfg.OUTPUT_DIR, "best.pt"), device)
        save_ur2(model, train_set, device, "train")
        if cfg.USE_VALIDATION:
            save_ur2(model, val_set, device, "val")

    print("=" * 70)
    print(f"Training finished | best epoch={best_epoch} | best HE->ST MSE={best_loss:.4f}")
    print(f"Results: {cfg.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()