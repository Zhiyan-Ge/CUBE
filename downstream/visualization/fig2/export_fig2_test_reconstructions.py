import csv
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_ROOT = "./model"
sys.path.insert(0, MODEL_ROOT)

from he_mihc.dataset import HEMIHCDataset
from he_mihc.evaluate_pearson import compute_batch_sample_pearson
from he_mihc.model import HEMIHCBranch

TEST_RAW_DIR = "./data/paired_data/he_mihc/raw_data/HEMIT_dataset/test"
HE_RAW_DIR = os.path.join(TEST_RAW_DIR, "input")
MIHC_RAW_DIR = os.path.join(TEST_RAW_DIR, "label")
TEST_PKL_DIR = "./data/train_data/final_data/test/pkl"

CUBE_CHECKPOINT = "./models/test_models_3/test11/he_mihc/best_pearson.pt"

RESULT_DIR = "./result/visualization/fig2/result"
OUTPUT_DIR = os.path.join(RESULT_DIR, "test_reconstructions")
METRICS_CSV = os.path.join(RESULT_DIR, "fig2_reconstruction_metrics.csv")

GPU_ID = 3
BATCH_SIZE = 4
NUM_WORKERS = 8
MIHC_GAMMA = 0.5


def read_rgb(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mihc_composite_array(x, gamma=MIHC_GAMMA):
    x = np.clip(x, 0, 1) ** gamma
    return np.stack([x[2], x[1], x[0]], axis=-1)


def mihc_composite_raw(image):
    x = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    return mihc_composite_array(x)


def mihc_composite_tensor(x):
    return mihc_composite_array(x.detach().float().cpu().numpy())


def he_tensor_to_rgb(x):
    return np.clip(x.detach().float().cpu().permute(1, 2, 0).numpy(), 0, 1)


def load_model(device):
    model = HEMIHCBranch(groups=8, marker_specific_mihc=True).to(device)
    checkpoint = torch.load(CUBE_CHECKPOINT, map_location="cpu")
    state = checkpoint["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def save_figure(name, mean_r, raw_he, raw_mihc, pred_mihc, pred_he):
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0))

    images = [
        (raw_he, "H&E"),
        (mihc_composite_raw(raw_mihc), "Measured mIHC"),
        (mihc_composite_tensor(pred_mihc), "H&E → mIHC"),
        (he_tensor_to_rgb(pred_he), "mIHC → H&E"),
    ]

    for ax, (image, title) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{name}   |   Mean marker Pearson = {mean_r:.3f}", fontsize=10)
    plt.tight_layout()

    filename = f"{name}_reconstruction.png"
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")

    dataset = HEMIHCDataset(TEST_PKL_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = load_model(device)
    rows = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Test reconstructions", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=True):
                outputs = model(he, mihc)

            pred_mihc = outputs["he_to_mihc"].float()
            pearson_rows = compute_batch_sample_pearson(
                pred_mihc, mihc.float(), batch["sample_name"]
            )

            for i, metric in enumerate(pearson_rows):
                name = batch["sample_name"][i]

                raw_he = read_rgb(os.path.join(HE_RAW_DIR, name + ".tif"))
                raw_mihc = read_rgb(os.path.join(MIHC_RAW_DIR, name + ".tif"))

                filename = save_figure(
                    name,
                    metric["average"],
                    raw_he,
                    raw_mihc,
                    outputs["he_to_mihc"][i],
                    outputs["mihc_to_he"][i],
                )

                rows.append({
                    "sample_name": name,
                    "DAPI": metric["DAPI"],
                    "CD3": metric["CD3"],
                    "panCK": metric["panCK"],
                    "average": metric["average"],
                    "image_file": filename,
                })

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_name", "DAPI", "CD3", "panCK", "average", "image_file"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()