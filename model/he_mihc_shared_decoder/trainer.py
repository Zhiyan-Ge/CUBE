import os
import json
import time
import random
import traceback
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset import HEMIHCDataset
from .model import HEMIHCBranch
from he_mihc_shared_decoder import config as cfg
from loss.recon_loss import he_recon_loss, mihc_recon_loss
from loss.alignment_loss import ur_alignment_loss
from loss.model_loss import model_loss, branch_loss


LOSS_NAMES = ["total", "he_he", "he_mihc", "mihc_he", "mihc_mihc", "ur1"]
CHANNEL_NAMES = ["DAPI", "CD3", "panCK"]


def set_seed(seed):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_samples(dataset, n, seed):
    """Randomly select n samples; use the full dataset when n is None."""
    if n is None or n >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def build_loader(pkl_dir, n_samples, shuffle, seed, sampler=None):
    """Build a DataLoader for the HE-mIHC branch."""
    dataset = select_samples(HEMIHCDataset(pkl_dir), n_samples, seed)
    persistent = cfg.NUM_WORKERS > 0
    if sampler is not None:
        return DataLoader(
            dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, sampler=sampler,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=persistent
        )
    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=persistent
    )


def paired_augment(he, mihc):
    """Apply paired spatial transforms to HE and mIHC images.

    he: (B,C,H,W), mihc: (B,3,H,W). Both outputs remain on the same device.
    """
    # Use Python's random module, which is seeded by set_seed().
    B = he.size(0)
    he_out = he.clone()
    mihc_out = mihc.clone()
    for i in range(B):
        # Rotate k times by 90 degrees.
        k = random.randint(0, 3)
        if k != 0:
            he_out[i] = torch.rot90(he_out[i], k, [-2, -1])
            mihc_out[i] = torch.rot90(mihc_out[i], k, [-2, -1])

        # Horizontal flip.
        if random.random() < 0.5:
            he_out[i] = torch.flip(he_out[i], [-1])
            mihc_out[i] = torch.flip(mihc_out[i], [-1])

        # Vertical flip.
        if random.random() < 0.5:
            he_out[i] = torch.flip(he_out[i], [-2])
            mihc_out[i] = torch.flip(mihc_out[i], [-2])

    return he_out, mihc_out


def compute_pearson_channel_mean(pred, target, eps=1e-8):
    """Compute per-channel mean Pearson across valid samples in a batch.

    This function is used only for validation metrics and never participates in
    the gradient graph.
    """
    if pred.shape[1] != 3 or target.shape[1] != 3:
        return {name: 0.0 for name in CHANNEL_NAMES}, 0.0

    channel_totals = {name: torch.zeros((), device=pred.device, dtype=torch.float32) for name in CHANNEL_NAMES}
    channel_counts = {name: torch.zeros((), device=pred.device, dtype=torch.int64) for name in CHANNEL_NAMES}

    for idx, name in enumerate(CHANNEL_NAMES):
        p = pred[:, idx].reshape(pred.size(0), -1)
        t = target[:, idx].reshape(target.size(0), -1)

        p_centered = p - p.mean(dim=1, keepdim=True)
        t_centered = t - t.mean(dim=1, keepdim=True)
        target_var = t_centered.square().sum(dim=1)
        valid = target_var > eps

        if valid.any():
            numerator = (p_centered * t_centered).sum(dim=1)
            denominator = torch.sqrt(p_centered.square().sum(dim=1) * target_var + eps)
            pearson = torch.zeros_like(target_var)
            pearson[valid] = numerator[valid] / denominator[valid]
            pearson = pearson.clamp(-1.0 + eps, 1.0 - eps)
            channel_totals[name] = pearson[valid].sum()
            channel_counts[name] = valid.sum().to(torch.int64)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(channel_totals[name], op=dist.ReduceOp.SUM)
            dist.all_reduce(channel_counts[name], op=dist.ReduceOp.SUM)

        if channel_counts[name].item() > 0:
            channel_totals[name] = channel_totals[name] / float(channel_counts[name].item())
        else:
            channel_totals[name] = torch.zeros((), device=pred.device, dtype=torch.float32)

    summary = {name: float(channel_totals[name].detach().cpu().item()) for name in CHANNEL_NAMES}
    average = float(np.mean([summary[name] for name in CHANNEL_NAMES]))
    return summary, average


def compute_validation_pearson(model, loader, device):
    """Compute validation Pearson summary for HE→mIHC only."""
    model.eval()
    channel_totals = {name: torch.zeros((), device=device, dtype=torch.float32) for name in CHANNEL_NAMES}
    channel_counts = {name: torch.zeros((), device=device, dtype=torch.int64) for name in CHANNEL_NAMES}

    with torch.inference_mode():
        for batch in loader:
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                outputs = model(he, mihc)

            pred = outputs["he_to_mihc"].float()
            target = mihc.float()

            for idx, name in enumerate(CHANNEL_NAMES):
                p = pred[:, idx].reshape(pred.size(0), -1)
                t = target[:, idx].reshape(target.size(0), -1)
                p_centered = p - p.mean(dim=1, keepdim=True)
                t_centered = t - t.mean(dim=1, keepdim=True)
                target_var = t_centered.square().sum(dim=1)
                valid = target_var > 1e-8

                if valid.any():
                    numerator = (p_centered * t_centered).sum(dim=1)
                    denominator = torch.sqrt(p_centered.square().sum(dim=1) * target_var + 1e-8)
                    pearson = torch.zeros_like(target_var)
                    pearson[valid] = numerator[valid] / denominator[valid]
                    pearson = pearson.clamp(-1.0 + 1e-8, 1.0 - 1e-8)
                    channel_totals[name] += pearson[valid].sum().to(torch.float32)
                    channel_counts[name] += valid.sum().to(torch.int64)

    if dist.is_available() and dist.is_initialized():
        for name in CHANNEL_NAMES:
            dist.all_reduce(channel_totals[name], op=dist.ReduceOp.SUM)
            dist.all_reduce(channel_counts[name], op=dist.ReduceOp.SUM)

    summary = {name: 0.0 for name in CHANNEL_NAMES}
    for name in CHANNEL_NAMES:
        if channel_counts[name].item() > 0:
            summary[name] = float((channel_totals[name] / float(channel_counts[name].item())).detach().cpu().item())
    average = float(np.mean([summary[name] for name in CHANNEL_NAMES]))
    return {"DAPI": summary["DAPI"], "CD3": summary["CD3"], "panCK": summary["panCK"], "average": average}


def compute_losses(outputs, he, mihc):
    """Compute four reconstruction losses, UR1 alignment, and the branch loss."""
    he_he = he_recon_loss(outputs["he_to_he"], he, cfg.ALPHA1)
    he_mihc = mihc_recon_loss(
        outputs["he_to_mihc"], mihc, cfg.ALPHA2,
        cfg.MIHC_FG_THRESHOLD, cfg.MIHC_FG_EXTRA_WEIGHT, cfg.MIHC_CHANNEL_WEIGHTS,
        pearson_weight=cfg.MIHC_PEARSON_WEIGHT,
        pearson_scales=cfg.MIHC_PEARSON_SCALES,
        pearson_scale_weights=cfg.MIHC_PEARSON_SCALE_WEIGHTS,
        cd3_dice_weight=cfg.CD3_DICE_WEIGHT
    )
    mihc_he = he_recon_loss(outputs["mihc_to_he"], he, cfg.ALPHA1)
    mihc_mihc = mihc_recon_loss(
        outputs["mihc_to_mihc"], mihc, cfg.ALPHA2,
        cfg.MIHC_FG_THRESHOLD, cfg.MIHC_FG_EXTRA_WEIGHT, cfg.MIHC_CHANNEL_WEIGHTS,
        pearson_weight=0.0,
        pearson_scales=None,
        pearson_scale_weights=None,
        cd3_dice_weight=0.0
    )
    ur1 = ur_alignment_loss(outputs["ur1_he"], outputs["ur1_mihc"])

    m1 = model_loss(he_he, he_mihc, cfg.BETA1)
    m2 = model_loss(mihc_mihc, mihc_he, cfg.BETA2)
    total = branch_loss(m1, m2, ur1, cfg.GAMMA1)

    return {
        "total": total, "he_he": he_he, "he_mihc": he_mihc,
        "mihc_he": mihc_he, "mihc_mihc": mihc_mihc, "ur1": ur1
    }


def save_curves(history):
    """Save total and component loss curves."""
    epochs = np.arange(1, len(history["train"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [x["total"] for x in history["train"]], label="Train")
    if cfg.USE_VALIDATION:
        plt.plot(epochs, [x["total"] for x in history["val"]], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HE-mIHC Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "total_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name in LOSS_NAMES[1:]:
        plt.plot(epochs, [x[name] for x in history["train"]], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HE-mIHC Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "loss_components.png"), dpi=200)
    plt.close()


def save_report(history, train_size, val_size, best_total, best_cross, best_pearson, elapsed):
    """Save the training configuration, per-epoch losses, and final results."""
    pearson_history = []
    for idx, entry in enumerate(history["val"], start=1):
        if "pearson" in entry:
            pearson_history.append({
                "epoch": idx,
                "DAPI": entry["pearson"]["DAPI"],
                "CD3": entry["pearson"]["CD3"],
                "panCK": entry["pearson"]["panCK"],
                "average": entry["pearson"]["average"],
            })

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
        "MIHC_PEARSON_WEIGHT": cfg.MIHC_PEARSON_WEIGHT,
        "MIHC_PEARSON_SCALES": cfg.MIHC_PEARSON_SCALES,
        "MIHC_PEARSON_SCALE_WEIGHTS": cfg.MIHC_PEARSON_SCALE_WEIGHTS,
        "CD3_DICE_WEIGHT": cfg.CD3_DICE_WEIGHT,
        "loss_weights": {
            "alpha1": cfg.ALPHA1, "alpha2": cfg.ALPHA2,
            "beta1": cfg.BETA1, "beta2": cfg.BETA2, "gamma1": cfg.GAMMA1,
            "MIHC_FG_THRESHOLD": cfg.MIHC_FG_THRESHOLD,
            "MIHC_FG_EXTRA_WEIGHT": cfg.MIHC_FG_EXTRA_WEIGHT,
            "MIHC_CHANNEL_WEIGHTS": cfg.MIHC_CHANNEL_WEIGHTS,
            "USE_PAIRED_AUG": cfg.USE_PAIRED_AUG
        },
        "best_total": {"epoch": best_total[0], "loss": best_total[1]},
        "best_cross": {"epoch": best_cross[0], "loss": best_cross[1]},
        "best_pearson": best_pearson,
        "best_metric": "best_cross",
        "final_train": history["train"][-1],
        "final_val": history["val"][-1] if cfg.USE_VALIDATION else None,
        "pearson_history": pearson_history,
        "history": history
    }

    with open(os.path.join(cfg.OUTPUT_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def run_epoch(model, loader, device, optimizer=None, scaler=None, desc="Train"):
    """Run one training or validation epoch and return mean losses."""
    training = optimizer is not None
    model.train(training)
    sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main = rank == 0
    pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True) if is_main else loader

    for batch in pbar:
        he = batch["he"].to(device, non_blocking=True)
        mihc = batch["mihc"].to(device, non_blocking=True)
        batch_size = he.size(0)
        if training and cfg.USE_PAIRED_AUG:
            he, mihc = paired_augment(he, mihc)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                outputs = model(he, mihc)
                losses = compute_losses(outputs, he, mihc)

            if training:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_values = {name: losses[name].detach().item() for name in LOSS_NAMES}

        for name in LOSS_NAMES:
            sums[name] += loss_values[name] * batch_size
        sample_count += batch_size

        if is_main:
            postfix = {
                "total": f"{loss_values['total']:.4f}",
                "HE": f"{loss_values['he_he']:.3f}",
                "mIHC": f"{loss_values['mihc_mihc']:.3f}",
                "UR": f"{loss_values['ur1']:.3f}",
            }
            pbar.set_postfix(**postfix)

    if is_distributed:
        tensor = torch.tensor(
            [sums[name] for name in LOSS_NAMES] + [sample_count], dtype=torch.double, device=device
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = tensor.cpu().numpy()
        total_samples = reduced[-1]
        return {name: float(reduced[i]) / float(total_samples) for i, name in enumerate(LOSS_NAMES)}

    return {name: sums[name] / sample_count for name in LOSS_NAMES}


def save_checkpoint(model, optimizer, epoch, path):
    """Save model and optimizer states."""
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {"epoch": epoch, "model": state, "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, path)


def load_checkpoint(model, path, device):
    """Load model weights for final UR1 extraction."""
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(torch.load(path, map_location=device)["model"])


def save_ur1(model, loader, device, split):
    """Save H&E-derived UR1 features from the best model."""
    save_dir = os.path.join(cfg.OUTPUT_DIR, "ur1_he", split)
    os.makedirs(save_dir, exist_ok=True)
    target = model.module if hasattr(model, "module") else model
    target.eval()
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main = rank == 0

    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"Save {split} UR1", unit="batch", dynamic_ncols=True) if is_main else loader
        for batch in iterator:
            he = batch["he"].to(device, non_blocking=True)
            ur1 = target.model1.encoder(he).cpu().numpy()

            for name, feature in zip(batch["sample_name"], ur1):
                np.save(os.path.join(save_dir, f"{name}.npy"), feature)

    if is_distributed:
        dist.barrier()


def ddp_worker(rank, gpu_ids, port):
    world_size = len(gpu_ids)
    physical_gpu = gpu_ids[rank]
    try:
        if world_size > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(physical_gpu)
        device = torch.device(f"cuda:{physical_gpu}")
        set_seed(cfg.SEED)
        torch.backends.cudnn.benchmark = True

        is_main = rank == 0
        if is_main:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            start_time = time.time()

        # Data (use DistributedSampler)
        train_dataset = select_samples(HEMIHCDataset(cfg.TRAIN_DIR), cfg.TRAIN_SAMPLES, cfg.SEED)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        persistent = cfg.NUM_WORKERS > 0
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=persistent)

        if cfg.USE_VALIDATION:
            val_dataset = select_samples(HEMIHCDataset(cfg.VAL_DIR), cfg.VAL_SAMPLES, cfg.SEED + 1)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, sampler=val_sampler,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=persistent)
        else:
            val_loader = None

        # Model
        model = HEMIHCBranch(cfg.GROUPS).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[physical_gpu])

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)
        save_model = lambda epoch, path: save_checkpoint(model, optimizer, epoch, path)

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset) if val_loader else 0
        history = {"train": [], "val": []}
        best_total_loss, best_total_epoch = float("inf"), 0
        best_cross_loss, best_cross_epoch = float("inf"), 0
        best_pearson_dict = None
        best_pearson_epoch = 0
        best_pearson_value = -1.0

        if is_main:
            print("=" * 70)
            print(f"HE-mIHC training | train={train_size} | validation={cfg.USE_VALIDATION}")
            print(f"GPU={cfg.GPU_IDS} | batch={cfg.BATCH_SIZE} | epochs={cfg.EPOCHS}")
            print("=" * 70)

        # Training
        for epoch in range(1, cfg.EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(
                model, train_loader, device, optimizer, scaler, desc=f"Epoch {epoch}/{cfg.EPOCHS} Train"
            )
            history["train"].append(train_loss)

            if cfg.USE_VALIDATION:
                val_sampler.set_epoch(epoch)
                val_loss = run_epoch(model, val_loader, device, desc=f"Epoch {epoch}/{cfg.EPOCHS} Val")
                val_pearson = compute_validation_pearson(model, val_loader, device)
                val_loss["pearson"] = val_pearson
                history["val"].append(val_loss)
                total_metric = val_loss["total"]
                cross_metric = val_loss["he_mihc"] + val_loss["mihc_he"]
                if is_main:
                    line = (
                        f"Epoch {epoch} | train={train_loss['total']:.4f} | val_total={total_metric:.4f} | "
                        f"val_cross={cross_metric:.4f} | R(DAPI)={val_pearson['DAPI']:.4f} | "
                        f"R(CD3)={val_pearson['CD3']:.4f} | R(panCK)={val_pearson['panCK']:.4f} | "
                        f"R(avg)={val_pearson['average']:.4f}"
                    )
                    print(line)
                if is_main and val_pearson["average"] > best_pearson_value:
                    best_pearson_value = val_pearson["average"]
                    best_pearson_epoch = epoch
                    best_pearson_dict = val_pearson.copy()
                    save_model(epoch, os.path.join(cfg.OUTPUT_DIR, "best_pearson.pt"))
            else:
                total_metric = train_loss["total"]
                cross_metric = train_loss["he_mihc"] + train_loss["mihc_he"]
                if is_main:
                    line = f"Epoch {epoch} | train_total={total_metric:.4f} | train_cross={cross_metric:.4f}"
                    print(line)

            if is_main and total_metric < best_total_loss:
                best_total_loss, best_total_epoch = total_metric, epoch
                save_model(epoch, os.path.join(cfg.OUTPUT_DIR, "best_total.pt"))

            if is_main and cross_metric < best_cross_loss:
                best_cross_loss, best_cross_epoch = cross_metric, epoch
                save_model(epoch, os.path.join(cfg.OUTPUT_DIR, "best_cross.pt"))
                save_model(epoch, os.path.join(cfg.OUTPUT_DIR, "best.pt"))

            if is_main and epoch % cfg.SAVE_EVERY == 0:
                save_model(epoch, os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pt"))

        if is_main:
            save_model(cfg.EPOCHS, os.path.join(cfg.OUTPUT_DIR, "last.pt"))
            save_curves(history)
            save_report(history, train_size, val_size, (best_total_epoch, best_total_loss), (best_cross_epoch, best_cross_loss),
                        {"epoch": best_pearson_epoch, "average": best_pearson_value, **(best_pearson_dict or {"DAPI": 0.0, "CD3": 0.0, "panCK": 0.0, "average": 0.0})},
                        time.time() - start_time)

        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        if cfg.SAVE_UR1:
            cross_path = os.path.join(cfg.OUTPUT_DIR, "best_cross.pt")
            if not os.path.exists(cross_path):
                cross_path = os.path.join(cfg.OUTPUT_DIR, "best.pt")
            load_checkpoint(model, cross_path, device)
            save_ur1(model, train_loader, device, "train")
            if cfg.USE_VALIDATION:
                save_ur1(model, val_loader, device, "val")

        if is_main:
            print("=" * 70)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main():
    gpu_ids = cfg.GPU_IDS
    world_size = len(gpu_ids)
    if world_size > 1:
        port = random.randint(12000, 20000)
        mp.spawn(ddp_worker, args=(gpu_ids, port), nprocs=world_size, join=True)
    else:
        # Run a single process on the configured GPU.
        ddp_worker(0, gpu_ids, random.randint(12000, 20000))


if __name__ == "__main__":
    main()