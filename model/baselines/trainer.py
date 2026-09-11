import json
import os
import random
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as cfg
from .dataset import HEMITPKLDataset
from .metrics import batch_pearson
from .model import BenchmarkModel, build_losses, discriminator_loss, generator_loss


CHANNEL_NAMES = ["DAPI", "CD3", "panCK"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(path, training):
    dataset = HEMITPKLDataset(path)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=training,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=training,  # avoids BatchNorm 1×1 failure on a final single sample
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def set_requires_grad(module, enabled):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_epoch(
    model,
    loader,
    losses,
    optimizer_g,
    optimizer_d,
    scaler_g,
    scaler_d,
    device,
    epoch,
):
    model.train()

    total_g = 0.0
    total_l1 = 0.0
    total_gan = 0.0
    total_d = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{cfg.EPOCHS} Train",
        dynamic_ncols=True,
    )

    for step, batch in enumerate(progress, 1):
        he = batch["he"].to(device, non_blocking=True)
        mihc = batch["mihc"].to(device, non_blocking=True)

        with autocast(
            device_type="cuda",
            enabled=cfg.USE_AMP,
        ):
            fake = model(he)

        # ====================================================
        # 1. Update Discriminator
        # ====================================================
        if model.use_gan:
            set_requires_grad(model.discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(
                device_type="cuda",
                enabled=cfg.USE_AMP,
            ):
                loss_d = discriminator_loss(
                    model,
                    losses,
                    he,
                    mihc,
                    fake,
                )

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            total_d += loss_d.item()

        # ====================================================
        # 2. Update Generator
        # ====================================================
        optimizer_g.zero_grad(set_to_none=True)

        if model.use_gan:
            set_requires_grad(model.discriminator, False)

        with autocast(
            device_type="cuda",
            enabled=cfg.USE_AMP,
        ):
            loss_g, parts = generator_loss(
                model,
                losses,
                he,
                mihc,
                fake,
            )

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        total_g += loss_g.item()
        total_l1 += parts["l1"].item()
        total_gan += parts["gan"].item()

        if step % cfg.LOG_EVERY == 0:
            progress.set_postfix(
                G=f"{total_g / step:.4f}",
                L1=f"{total_l1 / step:.4f}",
                D=(
                    f"{total_d / step:.4f}"
                    if model.use_gan
                    else "-"
                ),
            )

    n = len(loader)

    return {
        "generator": total_g / n,
        "l1": total_l1 / n,
        "gan": total_gan / n,
        "discriminator": (
            total_d / n if model.use_gan else 0.0
        ),
    }


@torch.no_grad()
def validate(model, loader, losses, device):
    model.eval()
    l1_sum = 0.0
    corr_sum = torch.zeros(3, dtype=torch.float64)
    sample_count = 0

    for batch in tqdm(loader, desc="Validation", dynamic_ncols=True, leave=False):
        he = batch["he"].to(device, non_blocking=True)
        mihc = batch["mihc"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            pred = model(he)
            l1 = losses["l1"](pred, mihc)

        corr = batch_pearson(pred, mihc).cpu().double()
        corr_sum += corr.sum(dim=0)
        sample_count += corr.shape[0]
        l1_sum += l1.item() * corr.shape[0]

    means = corr_sum / sample_count
    result = {CHANNEL_NAMES[i]: means[i].item() for i in range(3)}
    result["average"] = means.mean().item()
    result["l1"] = l1_sum / sample_count
    return result


def save_checkpoint(path, model, epoch, val):
    torch.save(
        {
            "benchmark": cfg.BENCHMARK,
            "epoch": epoch,
            "generator": model.generator.state_dict(),
            "val": val,
        },
        path,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 512×512 benchmark.")

    seed_everything(cfg.SEED)
    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")

    output_dir = os.path.join(cfg.OUTPUT_ROOT, cfg.BENCHMARK)
    os.makedirs(output_dir, exist_ok=True)

    train_loader = make_loader(cfg.TRAIN_DIR, training=True)
    val_loader = make_loader(cfg.VAL_DIR, training=False)

    model = BenchmarkModel(cfg.BENCHMARK).to(device)
    losses = build_losses(device)

    optimizer_g = Adam(model.generator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
    scheduler_g = StepLR(optimizer_g, step_size=cfg.LR_STEP_EPOCH, gamma=cfg.LR_GAMMA)

    optimizer_d = None
    scheduler_d = None
    if model.use_gan:
        optimizer_d = Adam(model.discriminator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
        scheduler_d = StepLR(optimizer_d, step_size=cfg.LR_STEP_EPOCH, gamma=cfg.LR_GAMMA)

    scaler_g = GradScaler("cuda", enabled=cfg.USE_AMP)
    scaler_d = GradScaler("cuda", enabled=cfg.USE_AMP)

    print("=" * 72)
    print(f"HEMIT 512 benchmark: {cfg.BENCHMARK}")
    print(f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
    print(f"GPU={cfg.GPU_ID} | batch={cfg.BATCH_SIZE} | epochs={cfg.EPOCHS} | lr={cfg.LR}")
    print("=" * 72)

    history = []
    best = {"epoch": 0, "average": float("-inf")}
    start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            losses,
            optimizer_g,
            optimizer_d,
            scaler_g,
            scaler_d,
            device,
            epoch,
        )
        val_stats = validate(model, val_loader, losses, device)

        scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        row = {
            "epoch": epoch,
            "lr": optimizer_g.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"DAPI={val_stats['DAPI']:.4f} "
            f"CD3={val_stats['CD3']:.4f} "
            f"panCK={val_stats['panCK']:.4f} "
            f"Avg={val_stats['average']:.4f}"
        )

        if val_stats["average"] > best["average"]:
            best = {"epoch": epoch, **val_stats}
            save_checkpoint(os.path.join(output_dir, "best_pearson.pt"), model, epoch, val_stats)

        if epoch % cfg.SAVE_EVERY == 0 or epoch == cfg.EPOCHS:
            save_checkpoint(os.path.join(output_dir, f"epoch_{epoch}.pt"), model, epoch, val_stats)

        with open(os.path.join(output_dir, "training_report.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark": cfg.BENCHMARK,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "best_pearson": best,
                    "elapsed_seconds": time.time() - start,
                    "history": history,
                },
                f,
                indent=2,
            )

    print("=" * 72)
    print("Best Pearson:", best)
    print("=" * 72)


if __name__ == "__main__":
    main()
