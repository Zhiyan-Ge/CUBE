import json
import os

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as cfg
from .dataset import HEMITPKLDataset
from .metrics import batch_pearson
from .model import BenchmarkModel


CHANNEL_NAMES = ["DAPI", "CD3", "panCK"]


def main():
    split = cfg.EVAL_SPLIT.lower()
    if split not in {"val", "test"}:
        raise ValueError("EVAL_SPLIT must be 'val' or 'test'.")

    pkl_dir = cfg.VAL_DIR if split == "val" else cfg.TEST_DIR
    checkpoint_path = os.path.join(cfg.OUTPUT_ROOT, cfg.BENCHMARK, "best_pearson.pt")
    output_path = os.path.join(cfg.OUTPUT_ROOT, cfg.BENCHMARK, f"{split}_pearson.json")

    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")

    dataset = HEMITPKLDataset(pkl_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    model = BenchmarkModel(cfg.BENCHMARK).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(checkpoint["generator"])
    model.eval()

    values = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {split}", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            mihc = batch["mihc"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=cfg.USE_AMP):
                pred = model(he)
            values.append(batch_pearson(pred, mihc).cpu())

    corr = torch.cat(values, dim=0)
    result = {
        "benchmark": cfg.BENCHMARK,
        "split": split,
        "checkpoint": "best_pearson.pt",
        "num_samples": len(dataset),
    }

    for i, name in enumerate(CHANNEL_NAMES):
        channel = corr[:, i]
        result[name] = {
            "mean": channel.mean().item(),
            "median": channel.median().item(),
            "std": channel.std(unbiased=False).item(),
        }

    sample_average = corr.mean(dim=1)
    result["average"] = {
        "mean": sample_average.mean().item(),
        "median": sample_average.median().item(),
        "std": sample_average.std(unbiased=False).item(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
