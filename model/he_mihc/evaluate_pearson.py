import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from he_mihc import config as cfg
from he_mihc.dataset import HEMIHCDataset
from he_mihc.model import HEMIHCBranch

CHANNEL_NAMES = ["DAPI", "CD3", "panCK"]
EVAL_BATCH_SIZE = 8


def make_loader(batch_size: int = EVAL_BATCH_SIZE):
    dataset = HEMIHCDataset(cfg.TEST_DIR)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )


def pearson_for_1d(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    eps = 1e-12
    if target.std() < eps or pred.std() < eps:
        return 0.0

    corr = np.corrcoef(target, pred)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def compute_sample_pearson(pred: torch.Tensor, target: torch.Tensor, sample_name: str):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    channel_scores = {}
    for c, channel_name in enumerate(CHANNEL_NAMES):
        score = pearson_for_1d(target_np[0, c], pred_np[0, c])
        channel_scores[channel_name] = score

    average = float(np.mean([channel_scores[name] for name in CHANNEL_NAMES]))
    return {
        "sample_name": sample_name,
        "DAPI": channel_scores["DAPI"],
        "CD3": channel_scores["CD3"],
        "panCK": channel_scores["panCK"],
        "average": average,
    }


def compute_batch_sample_pearson(pred: torch.Tensor, target: torch.Tensor, sample_names):
    batch_rows = []
    for b in range(pred.shape[0]):
        sample_name = sample_names[b]
        row = {
            "sample_name": sample_name,
        }
        channel_values = []
        for c, channel_name in enumerate(CHANNEL_NAMES):
            target_channel = target[b, c].detach().cpu().numpy().reshape(-1)
            pred_channel = pred[b, c].detach().cpu().numpy().reshape(-1)
            score = pearson_for_1d(target_channel, pred_channel)
            row[channel_name] = score
            channel_values.append(score)
        row["average"] = float(np.mean(channel_values))
        batch_rows.append(row)
    return batch_rows


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "DAPI", "CD3", "panCK", "average"])
        for row in rows:
            writer.writerow([
                row["sample_name"],
                row["DAPI"],
                row["CD3"],
                row["panCK"],
                row["average"],
            ])


def save_summary_json(path, checkpoint_name, rows):
    dapi_values = [row["DAPI"] for row in rows]
    cd3_values = [row["CD3"] for row in rows]
    panck_values = [row["panCK"] for row in rows]
    average_values = [row["average"] for row in rows]

    summary = {
        "checkpoint": checkpoint_name,
        "num_samples": len(rows),
        "DAPI": summarize(dapi_values),
        "CD3": summarize(cd3_values),
        "panCK": summarize(panck_values),
        "average": summarize(average_values),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_distribution_plot(path, rows):
    dapi_values = np.asarray([row["DAPI"] for row in rows], dtype=np.float64)
    cd3_values = np.asarray([row["CD3"] for row in rows], dtype=np.float64)
    panck_values = np.asarray([row["panCK"] for row in rows], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.hist(dapi_values, bins=30, alpha=0.7, label="DAPI")
    plt.hist(cd3_values, bins=30, alpha=0.7, label="CD3")
    plt.hist(panck_values, bins=30, alpha=0.7, label="panCK")
    plt.xlabel("Pearson correlation")
    plt.ylabel("count")
    plt.title("HE → mIHC Pearson distribution by channel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_summary(checkpoint_name, rows):
    dapi_values = [row["DAPI"] for row in rows]
    cd3_values = [row["CD3"] for row in rows]
    panck_values = [row["panCK"] for row in rows]
    average_values = [row["average"] for row in rows]

    print("\n" + "=" * 60)
    print(f"HE → mIHC Pearson | {checkpoint_name}")
    print("=" * 60)
    for label, values in [
        ("DAPI", dapi_values),
        ("CD3", cd3_values),
        ("panCK", panck_values),
        ("Average", average_values),
    ]:
        arr = np.asarray(values, dtype=np.float64)
        print(f"{label:<7s}: mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, std={np.std(arr):.4f}")
    print("=" * 60)


def load_model(checkpoint_path: str):
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Support the "module." prefix used by DDP checkpoints.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    # Infer the HE→mIHC decoder architecture from the checkpoint.
    marker_specific_mihc = any(key.startswith("model1.mihc_decoder.dapi_decoder.") for key in state_dict)

    model = HEMIHCBranch(cfg.GROUPS, marker_specific_mihc=marker_specific_mihc).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, device


def evaluate_checkpoint(checkpoint_name: str):
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loader = make_loader(EVAL_BATCH_SIZE)
    model, device = load_model(checkpoint_path)

    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {checkpoint_name}", unit="batch", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            sample_names = batch["sample_name"]

            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP and device.type == "cuda"):
                outputs = model(he, mihc)

            pred = outputs["he_to_mihc"].float()
            target = mihc.float()
            batch_rows = compute_batch_sample_pearson(pred, target, sample_names)
            rows.extend(batch_rows)

    output_dir = os.path.join(cfg.OUTPUT_DIR, "diagnostics", "pearson")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{checkpoint_name.replace('.pt', '')}_pearson.csv")
    summary_path = os.path.join(output_dir, f"{checkpoint_name.replace('.pt', '')}_pearson_summary.json")
    plot_path = os.path.join(output_dir, "pearson_distribution.png")

    save_csv(csv_path, rows)
    save_summary_json(summary_path, checkpoint_name, rows)
    save_distribution_plot(plot_path, rows)
    print_summary(checkpoint_name, rows)

    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {summary_path}")
    print(f"Plot saved: {plot_path}")
    return rows


def main():
    checkpoint_names = ["best_cross.pt", "best_pearson.pt", "best_total.pt", "last.pt", "epoch_80.pt"]

    available = [name for name in checkpoint_names if os.path.exists(os.path.join(cfg.OUTPUT_DIR, name))]
    if not available:
        raise FileNotFoundError(f"No checkpoint found in {cfg.OUTPUT_DIR}. Expected one of: {checkpoint_names}")

    primary = "best_pearson.pt"
    if primary in available:
        evaluate_checkpoint(primary)
        return

    for name in available:
        evaluate_checkpoint(name)
        break


if __name__ == "__main__":
    main()