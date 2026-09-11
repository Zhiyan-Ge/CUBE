#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# ============================================================
# User configuration
# ============================================================

HE_DIR = os.environ.get("CUBE_HE_DIR", "/path/to/HEMIT/input")
MIHC_DIR = os.environ.get("CUBE_MIHC_DIR", "/path/to/HEMIT/label")
OUTPUT_DIR = os.environ.get("CUBE_PREPROCESSING_OUTPUT", "./output/preprocessing")
REPORT_PATH = os.path.join(OUTPUT_DIR, "preprocessing_report.json")

MAX_SAMPLES = None
GPU_IDS = [3]

DEEPSPOT_REPO_PATH = os.environ.get("DEEPSPOT_REPO_PATH", "/path/to/DeepSpot")
DEEPSPOT_MODEL_WEIGHTS_PATH = os.environ.get("DEEPSPOT_MODEL_WEIGHTS_PATH", "/path/to/DeepSpot_model/final_model.pkl")
DEEPSPOT_MODEL_HPARAM_PATH = os.environ.get("DEEPSPOT_MODEL_HPARAM_PATH", "/path/to/DeepSpot_model/top_param_overall.yaml")
DEEPSPOT_GENE_INFO_CSV_PATH = os.environ.get("DEEPSPOT_GENE_INFO_CSV_PATH", "/path/to/DeepSpot_model/info_highly_variable_genes.csv")
DEEPSPOT_MORPHOLOGY_MODEL_PATH = os.environ.get("DEEPSPOT_MORPHOLOGY_MODEL_PATH", "/path/to/DeepSpot_model/pytorch_model.bin")

GRID_SIZE = 16
TARGET_GENE_COUNT = 256
SPOT_DIAMETER_PX = None
N_MINI_TILES = 9
NEIGHBOR_RADIUS = 1
FILTER_WHITE = False
WHITE_CUTOFF = 200.0
CLIP_NEGATIVE = False
ST_DTYPE = "float32"

TIMING_KEYS = ["read", "concept", "deepspot", "resize", "save", "total"]


# ============================================================
# Statistics
# ============================================================

def new_stats():
    return {"n": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}


def update_stats(s, x):
    x = np.asarray(x)
    s["n"] += x.size
    s["sum"] += float(np.sum(x, dtype=np.float64))
    s["sum_sq"] += float(np.sum(np.square(x, dtype=np.float64)))
    s["min"] = min(s["min"], float(np.min(x)))
    s["max"] = max(s["max"], float(np.max(x)))


def merge_stats(a, b):
    a["n"] += b["n"]
    a["sum"] += b["sum"]
    a["sum_sq"] += b["sum_sq"]
    a["min"] = min(a["min"], b["min"])
    a["max"] = max(a["max"], b["max"])


def finish_stats(s):
    mean = s["sum"] / s["n"]
    std = max(s["sum_sq"] / s["n"] - mean * mean, 0.0) ** 0.5
    return {"min": s["min"], "max": s["max"], "mean": mean, "std": std}


def new_vector_stats(dim):
    return {
        "n": 0,
        "sum": np.zeros(dim, dtype=np.float64),
        "sum_sq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf),
        "max": np.full(dim, -np.inf),
    }


def update_vector_stats(s, x):
    x = np.asarray(x, dtype=np.float64)
    s["n"] += 1
    s["sum"] += x
    s["sum_sq"] += x * x
    s["min"] = np.minimum(s["min"], x)
    s["max"] = np.maximum(s["max"], x)


def merge_vector_stats(a, b):
    a["n"] += b["n"]
    a["sum"] += b["sum"]
    a["sum_sq"] += b["sum_sq"]
    a["min"] = np.minimum(a["min"], b["min"])
    a["max"] = np.maximum(a["max"], b["max"])


def finish_vector_stats(s):
    mean = s["sum"] / s["n"]
    std = np.sqrt(np.maximum(s["sum_sq"] / s["n"] - mean * mean, 0.0))
    return {"min": s["min"].tolist(), "max": s["max"].tolist(), "mean": mean.tolist(), "std": std.tolist()}


def new_aggregate():
    return {
        "samples": 0,
        "he512": new_stats(),
        "mihc512": new_stats(),
        "he256": new_stats(),
        "st": new_stats(),
        "concept": new_vector_stats(3),
        "otsu": new_vector_stats(3),
        "mega_coord": new_vector_stats(2),
        "grid_coord": new_vector_stats(2),
        "timing": new_vector_stats(len(TIMING_KEYS)),
        "st_quantiles": new_vector_stats(3),
        "st_sample_mean": new_stats(),
        "st_sample_std": new_stats(),
        "st_negative": 0,
        "st_zero": 0,
        "st_total": 0,
        "gene_sum": np.zeros(TARGET_GENE_COUNT, dtype=np.float64),
        "gene_sum_sq": np.zeros(TARGET_GENE_COUNT, dtype=np.float64),
        "gene_count": 0,
        "gene_names": None,
    }


def merge_aggregate(a, b):
    a["samples"] += b["samples"]

    for k in ["he512", "mihc512", "he256", "st", "st_sample_mean", "st_sample_std"]:
        merge_stats(a[k], b[k])

    for k in ["concept", "otsu", "mega_coord", "grid_coord", "timing", "st_quantiles"]:
        merge_vector_stats(a[k], b[k])

    for k in ["st_negative", "st_zero", "st_total", "gene_count"]:
        a[k] += b[k]

    a["gene_sum"] += b["gene_sum"]
    a["gene_sum_sq"] += b["gene_sum_sq"]

    if a["gene_names"] is None:
        a["gene_names"] = b["gene_names"]


# ============================================================
# One worker per GPU
# ============================================================

def worker(gpu_id, filenames, queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch
        from _01_image_reading import process_single_paired_patch
        from _02_down_sample import prepare_cube_images
        from _03_concept_scorer import calculate_otsu_thresholds, calculate_patch_scores
        from _04_ST_processing import DeepSpotConfig, DeepSpotPseudoSTGenerator
        from _05_dict2pkl import save_sample_as_pickle

        torch.backends.cudnn.benchmark = True
        t_init = time.perf_counter()

        config = DeepSpotConfig(
            deepspot_repo_path=DEEPSPOT_REPO_PATH,
            model_weights_path=DEEPSPOT_MODEL_WEIGHTS_PATH,
            model_hparam_path=DEEPSPOT_MODEL_HPARAM_PATH,
            gene_info_csv_path=DEEPSPOT_GENE_INFO_CSV_PATH,
            morphology_model_path=DEEPSPOT_MORPHOLOGY_MODEL_PATH,
            device="cuda",
            grid_size=GRID_SIZE,
            target_gene_count=TARGET_GENE_COUNT,
            spot_diameter_px=SPOT_DIAMETER_PX,
            n_mini_tiles=N_MINI_TILES,
            neighbor_radius=NEIGHBOR_RADIUS,
            filter_white=FILTER_WHITE,
            white_cutoff=WHITE_CUTOFF,
            clip_negative=CLIP_NEGATIVE,
            dtype=ST_DTYPE,
        )

        generator = DeepSpotPseudoSTGenerator(config)
        queue.put(("ready", gpu_id, time.perf_counter() - t_init))

        agg = new_aggregate()

        for filename in filenames:
            try:
                he_path = os.path.join(HE_DIR, filename)
                mihc_path = os.path.join(MIHC_DIR, filename)
                sample_name = os.path.splitext(filename)[0]

                t0 = time.perf_counter()

                raw = process_single_paired_patch(he_path, mihc_path)
                he_1024 = raw["he_matrix_1024"]
                mihc_1024 = raw["mihc_matrix_1024"]

                t1 = time.perf_counter()

                thresholds = calculate_otsu_thresholds(mihc_1024)
                concept_scores = calculate_patch_scores(mihc_1024, thresholds)

                t2 = time.perf_counter()

                with torch.inference_mode():
                    st_matrix = generator.predict_from_image_path(he_path)

                t3 = time.perf_counter()

                he_512, mihc_512, he_256 = prepare_cube_images(he_1024, mihc_1024)

                t4 = time.perf_counter()

                save_sample_as_pickle(
                    normalized_mega_coord=raw["normalized_mega_coord"],
                    patch_grid_coord=raw["patch_grid_coord"],
                    he_matrix_512=he_512,
                    mihc_matrix_512=mihc_512,
                    he_matrix_256=he_256,
                    concept_scores=concept_scores,
                    st_matrix=st_matrix,
                    save_path=os.path.join(OUTPUT_DIR, sample_name + ".pkl"),
                    sample_name=sample_name,
                    metadata={},
                )

                t5 = time.perf_counter()

                update_stats(agg["he512"], he_512)
                update_stats(agg["mihc512"], mihc_512)
                update_stats(agg["he256"], he_256)
                update_stats(agg["st"], st_matrix)
                update_vector_stats(agg["concept"], concept_scores)
                update_vector_stats(agg["otsu"], thresholds)
                update_vector_stats(agg["mega_coord"], raw["normalized_mega_coord"])
                update_vector_stats(agg["grid_coord"], raw["patch_grid_coord"])
                update_vector_stats(agg["timing"], [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0])

                st_mean = float(np.mean(st_matrix))
                st_std = float(np.std(st_matrix))
                update_stats(agg["st_sample_mean"], [st_mean])
                update_stats(agg["st_sample_std"], [st_std])
                update_vector_stats(agg["st_quantiles"], np.quantile(st_matrix, [0.01, 0.50, 0.99]))

                agg["st_negative"] += int(np.sum(st_matrix < 0))
                agg["st_zero"] += int(np.sum(st_matrix == 0))
                agg["st_total"] += st_matrix.size

                flat = st_matrix.reshape(-1, TARGET_GENE_COUNT).astype(np.float64)
                agg["gene_sum"] += flat.sum(axis=0)
                agg["gene_sum_sq"] += np.square(flat).sum(axis=0)
                agg["gene_count"] += flat.shape[0]
                agg["samples"] += 1

                queue.put(("done", gpu_id, filename, t5 - t0, t3 - t2))

            except Exception as e:
                queue.put(("error", gpu_id, filename, repr(e)))

        agg["gene_names"] = generator.get_gene_names()
        queue.put(("finished", gpu_id, agg))

    except Exception as e:
        queue.put(("fatal", gpu_id, repr(e)))


# ============================================================
# Main workflow
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filenames = sorted(f for f in os.listdir(HE_DIR) if f.lower().endswith((".tif", ".tiff")))
    if MAX_SAMPLES is not None:
        filenames = filenames[:MAX_SAMPLES]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    shards = [filenames[i::len(GPU_IDS)] for i in range(len(GPU_IDS))]
    workers = []

    for gpu_id, shard in zip(GPU_IDS, shards):
        if shard:
            p = ctx.Process(target=worker, args=(gpu_id, shard, queue))
            p.start()
            workers.append(p)

    print(f"Samples: {len(filenames)} | GPUs: {GPU_IDS} | workers: {len(workers)}", flush=True)

    global_agg = new_aggregate()
    finished_workers = 0
    failed_samples = 0

    with tqdm(total=len(filenames), desc="CUBE preprocessing", unit="sample", dynamic_ncols=True) as pbar:
        while finished_workers < len(workers):
            msg = queue.get()
            kind = msg[0]

            if kind == "ready":
                _, gpu, sec = msg
                tqdm.write(f"[GPU {gpu}] DeepSpot ready: {sec:.1f}s")

            elif kind == "done":
                _, gpu, filename, total_sec, st_sec = msg
                pbar.update(1)
                pbar.set_postfix(GPU=gpu, total=f"{total_sec:.1f}s", ST=f"{st_sec:.1f}s")

            elif kind == "error":
                _, gpu, filename, error = msg
                failed_samples += 1
                pbar.update(1)
                tqdm.write(f"[GPU {gpu}] ERROR {filename}: {error}")

            elif kind == "finished":
                _, gpu, agg = msg
                merge_aggregate(global_agg, agg)
                finished_workers += 1

            elif kind == "fatal":
                _, gpu, error = msg
                for p in workers:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(f"GPU {gpu} worker fatal error: {error}")

    for p in workers:
        p.join()

    gene_mean = global_agg["gene_sum"] / global_agg["gene_count"]
    gene_var = np.maximum(global_agg["gene_sum_sq"] / global_agg["gene_count"] - gene_mean ** 2, 0.0)

    report = {
        "requested_samples": len(filenames),
        "processed_samples": global_agg["samples"],
        "failed_samples": failed_samples,
        "gpu_ids": GPU_IDS,
        "he_matrix_512": {"shape": [3, 512, 512], "dtype": "float16", **finish_stats(global_agg["he512"])},
        "mihc_matrix_512": {"shape": [3, 512, 512], "dtype": "float16", **finish_stats(global_agg["mihc512"])},
        "he_matrix_256": {"shape": [3, 256, 256], "dtype": "float16", **finish_stats(global_agg["he256"])},
        "concept_scores": finish_vector_stats(global_agg["concept"]),
        "otsu_thresholds": finish_vector_stats(global_agg["otsu"]),
        "normalized_mega_coord": finish_vector_stats(global_agg["mega_coord"]),
        "patch_grid_coord": finish_vector_stats(global_agg["grid_coord"]),
        "timing_seconds": {"fields": TIMING_KEYS, **finish_vector_stats(global_agg["timing"])},
        "pseudo_st": {
            "shape": [GRID_SIZE, GRID_SIZE, TARGET_GENE_COUNT],
            "dtype": ST_DTYPE,
            **finish_stats(global_agg["st"]),
            "negative_ratio": global_agg["st_negative"] / global_agg["st_total"],
            "zero_ratio": global_agg["st_zero"] / global_agg["st_total"],
            "sample_mean": finish_stats(global_agg["st_sample_mean"]),
            "sample_std": finish_stats(global_agg["st_sample_std"]),
            "sample_quantiles_q01_q50_q99": finish_vector_stats(global_agg["st_quantiles"]),
            "gene_names": global_agg["gene_names"],
            "gene_mean": gene_mean.tolist(),
            "gene_std": np.sqrt(gene_var).tolist(),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted: {global_agg['samples']} succeeded, {failed_samples} failed")
    print(f"PKL: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()