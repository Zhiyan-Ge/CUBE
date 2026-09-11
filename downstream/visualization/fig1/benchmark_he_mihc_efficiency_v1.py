import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast

MODEL_ROOT = "./model"
sys.path.insert(0, MODEL_ROOT)

from hemit_benchmark.model import BenchmarkModel
from hemit512 import config as hemit_cfg
from hemit512.network import build_hemit_generator
from he_mihc.model import HEMIHCBranch


TEST_PKL_DIR = Path("./data/train_data/final_data/test/pkl")
OUTPUT_DIR = Path("./result/visualization/fig1/result/efficiency")

GPU_ID = 3
BATCH_SIZES = [1, 8]
WARMUP_ITERS = 50
TIMED_ITERS = 100
N_REPEATS = 5
USE_AMP = True

MODEL_NAMES = [
    "U-Net",
    "ResNet",
    "pix2pix U-Net",
    "pix2pix ResNet",
    "CUBE",
    "HEMIT-512 adapted",
]


class CUBEHEtoMIHC(nn.Module):
    def __init__(self):
        super().__init__()
        branch = HEMIHCBranch(groups=8, marker_specific_mihc=True)
        self.encoder = branch.model1.encoder
        self.decoder = branch.model1.mihc_decoder

    def forward(self, he):
        return self.decoder(self.encoder(he))


def build_model(name):
    if name == "U-Net":
        return BenchmarkModel("unet").generator
    if name == "ResNet":
        return BenchmarkModel("resnet").generator
    if name == "pix2pix U-Net":
        return BenchmarkModel("pix2pix_unet").generator
    if name == "pix2pix ResNet":
        return BenchmarkModel("pix2pix_resnet").generator
    if name == "CUBE":
        return CUBEHEtoMIHC()
    if name == "HEMIT-512 adapted":
        return build_hemit_generator(hemit_cfg)
    raise ValueError(name)


def load_real_he(batch_size, device):
    import pickle

    paths = sorted(TEST_PKL_DIR.glob("*.pkl"))[:batch_size]
    images = []

    for path in paths:
        with open(path, "rb") as f:
            sample = pickle.load(f)
        images.append(torch.from_numpy(sample["he_matrix_512"]).float())

    return torch.stack(images, dim=0).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def forward_model(model, x):
    with autocast(device_type="cuda", enabled=USE_AMP):
        return model(x)


def benchmark_latency(model, x):
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            forward_model(model, x)

    torch.cuda.synchronize()

    run_ms = []

    for _ in range(N_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        with torch.inference_mode():
            for _ in range(TIMED_ITERS):
                forward_model(model, x)

        end.record()
        torch.cuda.synchronize()

        run_ms.append(start.elapsed_time(end) / TIMED_ITERS)

    return np.asarray(run_ms, dtype=np.float64)


def benchmark_memory(model, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(10):
            forward_model(model, x)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def benchmark_model(name, batch_size, device):
    gc.collect()
    torch.cuda.empty_cache()

    model = build_model(name).to(device).eval()
    x = load_real_he(batch_size, device)

    params = count_parameters(model)
    latency_runs = benchmark_latency(model, x)
    peak_memory = benchmark_memory(model, x)

    batch_latency_mean = latency_runs.mean()
    batch_latency_std = latency_runs.std(ddof=1)
    patch_latency_mean = batch_latency_mean / batch_size
    patch_latency_std = batch_latency_std / batch_size
    throughput = batch_size * 1000.0 / batch_latency_mean

    result = {
        "model": name,
        "batch_size": batch_size,
        "parameters": params,
        "parameters_million": params / 1e6,
        "batch_latency_ms_mean": batch_latency_mean,
        "batch_latency_ms_std": batch_latency_std,
        "latency_ms_per_patch": patch_latency_mean,
        "latency_ms_per_patch_std": patch_latency_std,
        "throughput_patches_per_s": throughput,
        "peak_gpu_memory_mb": peak_memory,
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "timing_repeats": N_REPEATS,
        "amp": USE_AMP,
    }

    print(
        f"{name:20s} | batch={batch_size:2d} | "
        f"params={params / 1e6:8.3f} M | "
        f"latency={patch_latency_mean:8.3f} ms/patch | "
        f"throughput={throughput:8.2f} patch/s | "
        f"memory={peak_memory:8.1f} MB"
    )

    del model, x
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")

    print("=" * 110)
    print("CUBE HE→mIHC EFFICIENCY BENCHMARK")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"AMP: {USE_AMP}")
    print(f"Warm-up: {WARMUP_ITERS}")
    print(f"Timed iterations: {TIMED_ITERS}")
    print(f"Repeats: {N_REPEATS}")
    print("=" * 110)

    rows = []

    for batch_size in BATCH_SIZES:
        print()
        print(f"BATCH SIZE = {batch_size}")
        print("-" * 110)

        for name in MODEL_NAMES:
            rows.append(benchmark_model(name, batch_size, device))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "he_mihc_efficiency_benchmark.csv", index=False)

    batch1 = results[results["batch_size"] == 1].copy()
    batch8 = results[results["batch_size"] == 8].copy()

    summary = batch1[
        ["model", "parameters_million", "latency_ms_per_patch", "latency_ms_per_patch_std", "peak_gpu_memory_mb"]
    ].merge(
        batch8[["model", "throughput_patches_per_s", "peak_gpu_memory_mb"]],
        on="model",
        suffixes=("_batch1", "_batch8"),
    )

    summary = summary.rename(columns={
        "parameters_million": "parameters_M",
        "latency_ms_per_patch": "latency_ms_patch_batch1",
        "latency_ms_per_patch_std": "latency_ms_patch_batch1_std",
        "throughput_patches_per_s": "throughput_patch_s_batch8",
        "peak_gpu_memory_mb_batch1": "peak_memory_MB_batch1",
        "peak_gpu_memory_mb_batch8": "peak_memory_MB_batch8",
    })

    summary.to_csv(OUTPUT_DIR / "he_mihc_efficiency_summary.csv", index=False)

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)
    print(summary.to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()