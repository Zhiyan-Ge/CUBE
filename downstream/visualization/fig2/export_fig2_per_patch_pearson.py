import csv
import os
import sys

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_ROOT = "./model"
sys.path.insert(0, MODEL_ROOT)

from hemit_benchmark import config as baseline_cfg
from hemit_benchmark.dataset import HEMITPKLDataset as BaselineDataset
from hemit_benchmark.metrics import batch_pearson as baseline_pearson
from hemit_benchmark.model import BenchmarkModel
from hemit512 import config as hemit_cfg
from hemit512.dataset import HEMITPKLDataset as HEMITDataset
from hemit512.metrics import batch_pearson as hemit_pearson
from hemit512.network import build_hemit_generator
from he_mihc.dataset import HEMIHCDataset
from he_mihc.evaluate_pearson import compute_batch_sample_pearson
from he_mihc.model import HEMIHCBranch

TEST_PKL_DIR = "./data/train_data/final_data/test/pkl"

PIX2PIX_CHECKPOINT = "./models/hemit_512_benchmarks/pix2pix_resnet/best_pearson.pt"
HEMIT_CHECKPOINT = "./models/hemit_512_benchmarks/hemit/epoch_80.pt"
CUBE_CHECKPOINT = "./models/test_models_3/test11/he_mihc/best_pearson.pt"

RESULT_DIR = "./result/visualization/fig2/result"
PER_PATCH_DIR = os.path.join(RESULT_DIR, "per_patch")

GPU_ID = 3
BATCH_SIZE = 2
NUM_WORKERS = 8


def save_tensor_csv(path, names, corr):
    corr = corr.numpy()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "DAPI", "CD3", "panCK", "average"])
        for name, values in zip(names, corr):
            writer.writerow([name, *values.tolist(), float(values.mean())])


def save_rows_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_name", "DAPI", "CD3", "panCK", "average"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_cube(device):
    dataset = HEMIHCDataset(TEST_PKL_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    model = HEMIHCBranch(groups=8, marker_specific_mihc=True).to(device)
    checkpoint = torch.load(CUBE_CHECKPOINT, map_location="cpu")
    state = checkpoint["model"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="CUBE", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=True):
                pred = model(he, mihc)["he_to_mihc"].float()
            rows.extend(compute_batch_sample_pearson(pred, mihc.float(), batch["sample_name"]))

    save_rows_csv(os.path.join(PER_PATCH_DIR, "fig2_cube_per_patch.csv"), rows)
    means = [sum(row[m] for row in rows) / len(rows) for m in ["DAPI", "CD3", "panCK", "average"]]
    print("CUBE:", means)

    del model
    torch.cuda.empty_cache()


def evaluate_pix2pix(device):
    dataset = BaselineDataset(TEST_PKL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    model = BenchmarkModel("pix2pix_resnet").to(device)
    checkpoint = torch.load(PIX2PIX_CHECKPOINT, map_location=device)
    model.generator.load_state_dict(checkpoint["generator"])
    model.eval()

    names, values = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="pix2pix-ResNet", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=baseline_cfg.USE_AMP):
                pred = model(he)
            names.extend(batch["sample_name"])
            values.append(baseline_pearson(pred, mihc).cpu())

    corr = torch.cat(values)
    save_tensor_csv(os.path.join(PER_PATCH_DIR, "fig2_pix2pix_resnet_per_patch.csv"), names, corr)
    print("pix2pix-ResNet:", corr.mean(0).tolist(), "mean =", corr.mean().item())

    del model
    torch.cuda.empty_cache()


def evaluate_hemit(device):
    dataset = HEMITDataset(TEST_PKL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    model = build_hemit_generator(hemit_cfg).to(device)
    checkpoint = torch.load(HEMIT_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["generator"], strict=True)
    model.eval()

    names, values = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="HEMIT-512", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            pred = model(he)
            names.extend(batch["sample_name"])
            values.append(hemit_pearson(pred, mihc).cpu())

    corr = torch.cat(values)
    save_tensor_csv(os.path.join(PER_PATCH_DIR, "fig2_hemit512_per_patch.csv"), names, corr)
    print("HEMIT-512:", corr.mean(0).tolist(), "mean =", corr.mean().item())

    del model
    torch.cuda.empty_cache()


def main():
    os.makedirs(PER_PATCH_DIR, exist_ok=True)
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")

    evaluate_cube(device)
    evaluate_pix2pix(device)
    evaluate_hemit(device)


if __name__ == "__main__":
    main()
