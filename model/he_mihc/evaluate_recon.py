import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from he_mihc.dataset import HEMIHCDataset
from he_mihc.model import HEMIHCBranch
from he_mihc import config as cfg


EVAL_BATCH_SIZE = 8
FG_THRESHOLD = 0.05

CHECKPOINTS = [
    ("epoch20", "epoch_20.pt"),
    ("epoch30", "epoch_30.pt"),
    ("best", "best.pt"),
    ("last", "last.pt"),
]


def ssim_per_image(x, y, window=11):
    """Compute per-image SSIM for diagnostic evaluation."""
    padding = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=padding)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window, stride=1, padding=padding) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=padding) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )

    return ssim.flatten(1).mean(1)


def make_loader(pkl_dir, batch_size):
    dataset = HEMIHCDataset(pkl_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0
    )


def compute_train_means():
    """Compute channel-wise HE and mIHC means from the training set."""
    loader = make_loader(cfg.TRAIN_DIR, EVAL_BATCH_SIZE)

    he_sum = torch.zeros(3, dtype=torch.float64)
    mihc_sum = torch.zeros(3, dtype=torch.float64)

    pixel_count = 0

    for batch in loader:
        he = batch["he"].double()
        mihc = batch["mihc"].double()

        he_sum += he.sum(dim=(0, 2, 3))
        mihc_sum += mihc.sum(dim=(0, 2, 3))

        pixel_count += he.shape[0] * he.shape[2] * he.shape[3]

    return (he_sum / pixel_count).float(), (mihc_sum / pixel_count).float()


def new_stats():
    return {
        "samples": 0,
        "l1_sum": 0.0,
        "ssim_sum": 0.0,
        "channel_l1_sum": [0.0, 0.0, 0.0],
        "channel_ssim_sum": [0.0, 0.0, 0.0],
        "fg_error_sum": [0.0, 0.0, 0.0],
        "fg_count": [0.0, 0.0, 0.0],
        "bg_error_sum": [0.0, 0.0, 0.0],
        "bg_count": [0.0, 0.0, 0.0],
    }


def update_stats(stats, pred, target, mihc=False):
    batch_size = target.shape[0]

    l1 = torch.abs(pred - target).flatten(1).mean(1)

    ssim = ssim_per_image(pred, target)

    stats["samples"] += batch_size
    stats["l1_sum"] += l1.sum().item()
    stats["ssim_sum"] += ssim.sum().item()

    if not mihc:
        return

    for channel in range(3):
        p = pred[:, channel:channel + 1]
        t = target[:, channel:channel + 1]

        channel_l1 = torch.abs(p - t).flatten(1).mean(1)

        channel_ssim = ssim_per_image(p, t)

        stats["channel_l1_sum"][channel] += channel_l1.sum().item()
        stats["channel_ssim_sum"][channel] += channel_ssim.sum().item()

        error = torch.abs(p - t)

        fg = t > FG_THRESHOLD
        bg = ~fg

        stats["fg_error_sum"][channel] += error[fg].sum().item()
        stats["fg_count"][channel] += fg.sum().item()
        stats["bg_error_sum"][channel] += error[bg].sum().item()
        stats["bg_count"][channel] += bg.sum().item()


def finalize(stats):
    n = stats["samples"]

    result = {"l1": stats["l1_sum"] / n, "ssim": stats["ssim_sum"] / n}

    if sum(stats["channel_l1_sum"]) == 0:
        return result

    result["channels"] = {}

    for channel in range(3):
        fg_count = stats["fg_count"][channel]
        bg_count = stats["bg_count"][channel]

        result["channels"][f"channel_{channel}"] = {
            "l1": stats["channel_l1_sum"][channel] / n,
            "ssim": stats["channel_ssim_sum"][channel] / n,
            "foreground_l1": stats["fg_error_sum"][channel] / fg_count if fg_count > 0 else None,
            "background_l1": stats["bg_error_sum"][channel] / bg_count if bg_count > 0 else None,
            "foreground_fraction": fg_count / (fg_count + bg_count)
        }

    return result


def evaluate_baselines(loader, device, mean_he, mean_mihc):
    zero_mihc_stats = new_stats()
    mean_mihc_stats = new_stats()
    mean_he_stats = new_stats()

    mean_he = mean_he.to(device).view(1, 3, 1, 1)
    mean_mihc = mean_mihc.to(device).view(1, 3, 1, 1)

    with torch.inference_mode():
        for batch in loader:
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)

            zero_mihc = torch.zeros_like(mihc)

            mean_mihc_pred = mean_mihc.expand_as(mihc)
            mean_he_pred = mean_he.expand_as(he)

            update_stats(zero_mihc_stats, zero_mihc, mihc, mihc=True)
            update_stats(mean_mihc_stats, mean_mihc_pred, mihc, mihc=True)
            update_stats(mean_he_stats, mean_he_pred, he, mihc=False)

    return {
        "zero_mihc": finalize(zero_mihc_stats),
        "mean_mihc": finalize(mean_mihc_stats),
        "mean_he": finalize(mean_he_stats)
    }


def load_model(path, device):
    model = HEMIHCBranch(cfg.GROUPS).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_model(model, loader, device):
    stats = {
        "he_to_he": new_stats(),
        "he_to_mihc": new_stats(),
        "mihc_to_he": new_stats(),
        "mihc_to_mihc": new_stats()
    }

    with torch.inference_mode():
        for batch in loader:
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                outputs = model(he, mihc)

            update_stats(stats["he_to_he"], outputs["he_to_he"].float(), he, mihc=False)
            update_stats(stats["he_to_mihc"], outputs["he_to_mihc"].float(), mihc, mihc=True)
            update_stats(stats["mihc_to_he"], outputs["mihc_to_he"].float(), he, mihc=False)
            update_stats(stats["mihc_to_mihc"], outputs["mihc_to_mihc"].float(), mihc, mihc=True)

    return {name: finalize(value) for name, value in stats.items()}


def print_summary(results):
    print("\n" + "=" * 70)
    print("HE-mIHC diagnostic evaluation")
    print("=" * 70)

    baselines = results["baselines"]

    print("\nBaselines")
    print(
        "zero mIHC : "
        f"L1={baselines['zero_mihc']['l1']:.4f} | "
        f"SSIM={baselines['zero_mihc']['ssim']:.4f}"
    )

    print(
        "mean mIHC : "
        f"L1={baselines['mean_mihc']['l1']:.4f} | "
        f"SSIM={baselines['mean_mihc']['ssim']:.4f}"
    )

    print(
        "mean HE   : "
        f"L1={baselines['mean_he']['l1']:.4f} | "
        f"SSIM={baselines['mean_he']['ssim']:.4f}"
    )

    for label, values in results["checkpoints"].items():
        print("\n" + "-" * 70)
        print(label)

        for path_name, metrics in values.items():
            print(
                f"{path_name:15s} | "
                f"L1={metrics['l1']:.4f} | "
                f"SSIM={metrics['ssim']:.4f}"
            )

        channels = values["he_to_mihc"]["channels"]

        print("HE→mIHC channels:")

        for channel, metrics in channels.items():
            print(
                f"  {channel}: "
                f"L1={metrics['l1']:.4f}, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"FG-L1={metrics['foreground_l1']:.4f}, "
                f"BG-L1={metrics['background_l1']:.4f}, "
                f"FG={metrics['foreground_fraction']:.3f}"
            )


def main():
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(cfg.OUTPUT_DIR, "diagnostics")
    os.makedirs(save_dir, exist_ok=True)

    print("Computing train-set channel means...")
    mean_he, mean_mihc = compute_train_means()

    print("HE mean:", mean_he.tolist())
    print("mIHC mean:", mean_mihc.tolist())

    val_loader = make_loader(cfg.VAL_DIR, EVAL_BATCH_SIZE)

    print("Evaluating trivial baselines...")

    results = {
        "foreground_threshold": FG_THRESHOLD,
        "train_mean_he": mean_he.tolist(),
        "train_mean_mihc": mean_mihc.tolist(),
        "baselines": evaluate_baselines(val_loader, device, mean_he, mean_mihc),
        "checkpoints": {}
    }

    for label, filename in CHECKPOINTS:
        path = os.path.join(cfg.OUTPUT_DIR, filename)

        if not os.path.exists(path):
            continue

        print(f"Evaluating {filename}...")

        model = load_model(path, device)
        results["checkpoints"][label] = evaluate_model(model, val_loader, device)

        del model
        torch.cuda.empty_cache()

    output_path = os.path.join(save_dir, "diagnostic_metrics.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_summary(results)

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()