#!/usr/bin/env python3
"""Evaluate frozen DeepSpot on the fixed Visium HD spatial test region."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_visium_hd_deepspot as core


# This file is produced by the decoder-only fine-tuning experiment.
SPATIAL_SPLIT_CSV = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_cube_finetune/decoder_only_100/finetuning_patch_split.csv"
)
OUT_DIR = Path(
    "./result/01_real_st/"
    "data/HD/visium_hd_deepspot_spatial_test"
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.OUT_DIR = OUT_DIR

    split = pd.read_csv(SPATIAL_SPLIT_CSV)
    test_metadata = split.loc[split["split"] == "test"].copy()
    test_metadata = test_metadata.sort_values(["block_row", "block_col"]).reset_index(drop=True)
    patch_indices = np.arange(len(test_metadata))
    gene_mapping = pd.read_csv(core.GENE_MAPPING).sort_values("model_channel").reset_index(drop=True)

    module = core.load_cube_deepspot_wrapper()
    generator = core.make_generator(module)
    expected_genes = gene_mapping["gene_name"].astype(str).tolist()
    if generator.get_gene_names() != expected_genes:
        raise ValueError("DeepSpot output gene order differs from gene_mapping.csv")

    order = test_metadata[["patch_id", "block_row", "block_col", "filtered_bins", "split"]].copy()
    order["normalized_patch"] = [
        str(core.NORMALIZED_PATCH_DIR / f"{patch_id}.png")
        for patch_id in order["patch_id"]
    ]
    order["st_target"] = [
        str(core.ST_TARGET_DIR / f"{patch_id}.npz")
        for patch_id in order["patch_id"]
    ]
    order.to_csv(OUT_DIR / "spatial_test_patch_order.csv", index=False)

    predictions, completed = core.run_domain(
        generator, test_metadata, "normalized",
        core.NORMALIZED_PATCH_DIR, patch_indices,
    )
    completed_indices = patch_indices[completed]
    truth, bin_masks, target_gene_mask = core.load_truth_and_masks(
        test_metadata, completed_indices
    )
    mapping_gene_mask = core.as_bool(gene_mapping["in_visium_hd"])
    if not np.array_equal(target_gene_mask, mapping_gene_mask):
        raise ValueError("gene_mask in ST targets differs from gene_mapping.csv")

    summaries = core.evaluate_domain(
        "normalized", predictions, truth, completed_indices,
        bin_masks, target_gene_mask, gene_mapping,
    )
    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "deepspot_spatial_test_summary.csv", index=False
    )

    report = {
        "method": "frozen DeepSpot direct evaluation on fixed spatial test region",
        "spatial_split_csv": str(SPATIAL_SPLIT_CSV),
        "split": "test",
        "input_domain": "Macenko + global Lab L* normalized",
        "pooling": "none",
        "grid": [16, 16],
        "input_patch_size_px": [1024, 1024],
        "bin_size_um": 16.0,
        "bin_size_px": 64,
        "target_normalization": "log1p(count / all-GEX library size * 10000)",
        "patches_in_spatial_split": int(len(split)),
        "test_patches_selected": int(len(test_metadata)),
        "test_patches_completed": int(completed.sum()),
        "valid_bins": int(truth.shape[0]),
        "genes_total": 256,
        "genes_evaluated": int(target_gene_mask.sum()),
        "missing_genes": gene_mapping.loc[
            ~mapping_gene_mask, "gene_name"
        ].astype(str).tolist(),
        "prediction_file_stores_unclipped_values": True,
        "deepspot": generator.describe(),
        "results": summaries,
    }
    with (OUT_DIR / "deepspot_spatial_test_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD DEEPSPOT SPATIAL TEST COMPLETE")
    print("=" * 80)
    print(f"test patches: {completed.sum()}/{len(test_metadata)}")
    print(f"valid 16-um bins: {truth.shape[0]}")
    print(f"genes evaluated: {target_gene_mask.sum()}/256")
    for row in summaries:
        print(
            f"{row['variant']:12s} | "
            f"gene Pearson={row['gene_pearson_mean_detected_ge_50']:.4f} | "
            f"gene Spearman={row['gene_spearman_mean_detected_ge_50']:.4f} | "
            f"bin Pearson={row['bin_pearson_mean']:.4f}"
        )
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()
