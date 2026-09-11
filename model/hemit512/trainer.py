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
from .network import GANLoss, build_discriminator, build_hemit_generator


CHANNEL_NAMES = ["DAPI", "CD3", "panCK"]


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        finite = tensor.detach()[torch.isfinite(tensor.detach())]
        finite_min = finite.min().item() if finite.numel() else float("nan")
        finite_max = finite.max().item() if finite.numel() else float("nan")
        raise FloatingPointError(
            f"Non-finite values detected in {name}: "
            f"shape={tuple(tensor.shape)}, finite_min={finite_min}, finite_max={finite_max}"
        )


def assert_finite_gradients(module, name):
    """Fail only when an individual gradient tensor actually contains NaN/Inf.

    Do NOT use clip_grad_norm_(..., max_norm=inf) as a finite check: its global
    FP32 norm reduction can itself become non-finite for very large but finite
    gradients, which makes the diagnostic ambiguous.
    """
    for param_name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            finite = grad.detach()[torch.isfinite(grad.detach())]
            finite_min = finite.min().item() if finite.numel() else float("nan")
            finite_max = finite.max().item() if finite.numel() else float("nan")
            raise FloatingPointError(
                f"Non-finite gradient in {name}.{param_name}: "
                f"shape={tuple(grad.shape)}, finite_min={finite_min}, finite_max={finite_max}"
            )


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
        drop_last=training,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def set_requires_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)


def train_epoch(generator, discriminator, loader, l1_loss, gan_loss,
                optimizer_g, optimizer_d, scaler_g, scaler_d, device, epoch):
    generator.train()
    discriminator.train()

    total_g = 0.0
    total_l1 = 0.0
    total_gan = 0.0
    total_d = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} Train", dynamic_ncols=True)

    for step, batch in enumerate(progress, 1):
        he = batch["he"].to(device, non_blocking=True)
        real_mihc = batch["mihc"].to(device, non_blocking=True)

        # ----------------------------------------------------
        # Generator forward ONCE, matching pix2pix optimize flow.
        # ----------------------------------------------------
        assert_finite("HE input", he)
        assert_finite("mIHC target", real_mihc)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            fake_mihc = generator(he)
        assert_finite("generator output", fake_mihc)

        # ----------------------------------------------------
        # 1) Discriminator update
        # ----------------------------------------------------
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            fake_pair = torch.cat([he, fake_mihc.detach()], dim=1)
            real_pair = torch.cat([he, real_mihc], dim=1)
            d_fake = gan_loss(discriminator(fake_pair), False)
            d_real = gan_loss(discriminator(real_pair), True)
            loss_d = 0.5 * (d_fake + d_real)
        assert_finite("D loss", loss_d)

        scaler_d.scale(loss_d).backward()
        if not cfg.USE_AMP:
            assert_finite_gradients(discriminator, "discriminator")
        scaler_d.step(optimizer_d)
        scaler_d.update()

        # ----------------------------------------------------
        # 2) Generator update
        # ----------------------------------------------------
        set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            fake_pair = torch.cat([he, fake_mihc], dim=1)
            loss_g_gan = gan_loss(discriminator(fake_pair), True)
            loss_l1 = l1_loss(fake_mihc, real_mihc)
            loss_g = loss_g_gan + cfg.LAMBDA_L1 * loss_l1
        assert_finite("G GAN loss", loss_g_gan)
        assert_finite("G L1 loss", loss_l1)
        assert_finite("G total loss", loss_g)

        scaler_g.scale(loss_g).backward()
        if not cfg.USE_AMP:
            assert_finite_gradients(generator, "generator")
        scaler_g.step(optimizer_g)
        scaler_g.update()

        total_g += loss_g.item()
        total_l1 += loss_l1.item()
        total_gan += loss_g_gan.item()
        total_d += loss_d.item()

        if step % cfg.LOG_EVERY == 0:
            progress.set_postfix(
                G=f"{total_g / step:.4f}",
                L1=f"{total_l1 / step:.4f}",
                GAN=f"{total_gan / step:.4f}",
                D=f"{total_d / step:.4f}",
            )

    n = len(loader)
    return {
        "generator": total_g / n,
        "l1": total_l1 / n,
        "gan": total_gan / n,
        "discriminator": total_d / n,
    }


@torch.no_grad()
def validate(generator, loader, l1_loss, device):
    generator.eval()
    l1_sum = 0.0
    corr_sum = torch.zeros(3, dtype=torch.float64)
    sample_count = 0

    for batch in tqdm(loader, desc="Validation", dynamic_ncols=True, leave=False):
        he = batch["he"].to(device, non_blocking=True)
        real_mihc = batch["mihc"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            pred = generator(he)
            l1 = l1_loss(pred, real_mihc)
        assert_finite("validation prediction", pred)
        assert_finite("validation L1", l1)

        corr = batch_pearson(pred, real_mihc).cpu().double()
        corr_sum += corr.sum(dim=0)
        sample_count += corr.shape[0]
        l1_sum += l1.item() * corr.shape[0]

    means = corr_sum / sample_count
    result = {CHANNEL_NAMES[i]: means[i].item() for i in range(3)}
    result["average"] = means.mean().item()
    result["l1"] = l1_sum / sample_count
    return result


def save_checkpoint(path, generator, discriminator, epoch, val):
    torch.save(
        {
            "benchmark": "hemit_512",
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "val": val,
            "config": {
                "img_size": cfg.IMG_SIZE,
                "patch_size": cfg.PATCH_SIZE,
                "window_size": cfg.WINDOW_SIZE,
                "lambda_l1": cfg.LAMBDA_L1,
                "lr": cfg.LR,
                "batch_size": cfg.BATCH_SIZE,
            },
        },
        path,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HEMIT-512 training.")

    seed_everything(cfg.SEED)
    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_loader = make_loader(cfg.TRAIN_DIR, True)
    val_loader = make_loader(cfg.VAL_DIR, False)

    generator = build_hemit_generator(cfg).to(device)
    discriminator = build_discriminator(cfg).to(device)

    l1_loss = torch.nn.L1Loss().to(device)
    gan_loss = GANLoss(cfg.GAN_MODE).to(device)

    optimizer_g = Adam(generator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
    scheduler_g = StepLR(optimizer_g, step_size=cfg.LR_STEP_EPOCH, gamma=cfg.LR_GAMMA)
    scheduler_d = StepLR(optimizer_d, step_size=cfg.LR_STEP_EPOCH, gamma=cfg.LR_GAMMA)

    scaler_g = GradScaler("cuda", enabled=cfg.USE_AMP)
    scaler_d = GradScaler("cuda", enabled=cfg.USE_AMP)

    print("=" * 72)
    print("HEMIT-512 adapted benchmark")
    print(f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
    print(
        f"GPU={cfg.GPU_ID} | batch={cfg.BATCH_SIZE} | epochs={cfg.EPOCHS} | "
        f"lr={cfg.LR} | lambda_L1={cfg.LAMBDA_L1}"
    )
    print("=" * 72)

    history = []
    best = {"epoch": 0, "average": float("-inf")}
    start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        train_stats = train_epoch(
            generator,
            discriminator,
            train_loader,
            l1_loss,
            gan_loss,
            optimizer_g,
            optimizer_d,
            scaler_g,
            scaler_d,
            device,
            epoch,
        )
        val_stats = validate(generator, val_loader, l1_loss, device)

        scheduler_g.step()
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
            save_checkpoint(
                os.path.join(cfg.OUTPUT_DIR, "best_pearson.pt"),
                generator,
                discriminator,
                epoch,
                val_stats,
            )

        if epoch % cfg.SAVE_EVERY == 0 or epoch == cfg.EPOCHS:
            save_checkpoint(
                os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pt"),
                generator,
                discriminator,
                epoch,
                val_stats,
            )

        with open(
            os.path.join(cfg.OUTPUT_DIR, "training_report.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "benchmark": "hemit_512",
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
