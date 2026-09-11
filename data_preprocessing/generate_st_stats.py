#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate training-set gene-wise statistics for CUBE pseudo-ST normalization."""

import argparse
import os
import pickle

import numpy as np


EXPECTED_ST_SHAPE = (16, 16, 256)
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate st_stats.npz from training-set CUBE pickle files."
    )
    parser.add_argument("train_pkl_dir", help="Directory containing training-set .pkl files.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <train_pkl_dir>/st_stats.npz.",
    )
    return parser.parse_args()


def generate_st_stats(train_pkl_dir: str, output_path: str, eps: float = EPS):
    pkl_paths = sorted(
        os.path.join(train_pkl_dir, name)
        for name in os.listdir(train_pkl_dir)
        if name.endswith(".pkl")
    )
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found in: {train_pkl_dir}")

    gene_sum = np.zeros(EXPECTED_ST_SHAPE[-1], dtype=np.float64)
    gene_sum_sq = np.zeros(EXPECTED_ST_SHAPE[-1], dtype=np.float64)
    total_spots = 0

    for index, pkl_path in enumerate(pkl_paths, start=1):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "st_matrix" not in data:
            raise KeyError(f"Missing st_matrix in: {pkl_path}")

        st = np.asarray(data["st_matrix"])
        if st.shape != EXPECTED_ST_SHAPE:
            raise ValueError(
                f"Unexpected st_matrix shape in {pkl_path}: "
                f"{st.shape}, expected {EXPECTED_ST_SHAPE}"
            )
        if not np.isfinite(st).all():
            raise ValueError(f"st_matrix contains NaN or Inf: {pkl_path}")

        flat = st.reshape(-1, EXPECTED_ST_SHAPE[-1]).astype(np.float64)
        gene_sum += flat.sum(axis=0)
        gene_sum_sq += np.square(flat).sum(axis=0)
        total_spots += flat.shape[0]

        if index % 100 == 0 or index == len(pkl_paths):
            print(f"Processed {index}/{len(pkl_paths)} files", flush=True)

    mean = gene_sum / total_spots
    variance = np.maximum(gene_sum_sq / total_spots - mean ** 2, 0.0)
    raw_std = np.sqrt(variance)
    safe_std = np.where(raw_std < eps, 1.0, raw_std)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        output_path,
        mean=mean.astype(np.float32),
        raw_std=raw_std.astype(np.float32),
        safe_std=safe_std.astype(np.float32),
        total_spots=np.int64(total_spots),
        num_files=np.int64(len(pkl_paths)),
        eps=np.float32(eps),
    )

    print(f"Saved: {output_path}")
    print(f"Files: {len(pkl_paths)}")
    print(f"Total spots: {total_spots}")


def main():
    args = parse_args()
    output_path = args.output or os.path.join(args.train_pkl_dir, "st_stats.npz")
    generate_st_stats(args.train_pkl_dir, output_path)


if __name__ == "__main__":
    main()
