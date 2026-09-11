#!/usr/bin/env python3
"""Compute and cache patch-level KEGG program scores for Fig. 5D."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SPATIAL_SPLIT = Path("./result/01_real_st/data/HD/visium_hd_cube_finetune/prepared/spatial_split.csv")
ST_TARGET_DIR = Path("./result/01_real_st/data/HD/visium_hd_16um_paired/st_targets")
GENE_LIST = Path("./result/05_real_st_program/data/program_definition/fig5d_kegg_program_gene_list.csv")
OUT_DIR = Path("./result/05_real_st_program/data/program_scores")

PROGRAM_COLUMNS = {
    "ECM_receptor_interaction": "ecm_receptor_score",
    "Complement_and_coagulation_cascades": "complement_coagulation_score",
}


def load_patch_expression(patch_id):
    with np.load(ST_TARGET_DIR / f"{patch_id}.npz") as target:
        values = target["st_log1p_cp10k"][target["bin_mask"].astype(bool)]
    return values.mean(axis=0).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SPATIAL_SPLIT).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    gene_list = pd.read_csv(GENE_LIST)
    patch_ids = metadata["patch_id"].astype(str).tolist()

    expression = np.stack([
        load_patch_expression(patch_id)
        for patch_id in tqdm(patch_ids, desc="Aggregate real ST", dynamic_ncols=True)
    ])

    train_mask = metadata["split"].eq("train").to_numpy()
    train_mean = expression[train_mask].mean(axis=0)
    train_std = expression[train_mask].std(axis=0)
    safe_std = np.where(train_std > 1e-6, train_std, 1.0)
    z_expression = (expression - train_mean) / safe_std

    scores = metadata.copy()
    for program, score_column in PROGRAM_COLUMNS.items():
        channels = gene_list.loc[
            gene_list["program"] == program, "model_channel"
        ].to_numpy(dtype=int)
        scores[score_column] = z_expression[:, channels].mean(axis=1)

    scores.to_csv(OUT_DIR / "fig5d_program_scores.csv", index=False)

    stats = gene_list.copy()
    channels = stats["model_channel"].to_numpy(dtype=int)
    stats["train_patch_mean"] = train_mean[channels]
    stats["train_patch_std"] = train_std[channels]
    stats["train_patch_safe_std"] = safe_std[channels]
    stats.to_csv(OUT_DIR / "fig5d_program_gene_stats.csv", index=False)

    np.savez_compressed(
        OUT_DIR / "fig5d_program_scores.npz",
        patch_id=scores["patch_id"].astype(str).to_numpy(),
        split=scores["split"].astype(str).to_numpy(),
        ecm_receptor_score=scores["ecm_receptor_score"].to_numpy(dtype=np.float32),
        complement_coagulation_score=scores["complement_coagulation_score"].to_numpy(dtype=np.float32),
        train_gene_mean=train_mean.astype(np.float32),
        train_gene_std=train_std.astype(np.float32),
        train_gene_safe_std=safe_std.astype(np.float32),
    )

    print(scores.groupby("split")[list(PROGRAM_COLUMNS.values())].agg(["count", "mean", "std"]))
    print(f"saved: {OUT_DIR / 'fig5d_program_scores.csv'}")
    print(f"saved: {OUT_DIR / 'fig5d_program_gene_stats.csv'}")
    print(f"saved: {OUT_DIR / 'fig5d_program_scores.npz'}")


if __name__ == "__main__":
    main()