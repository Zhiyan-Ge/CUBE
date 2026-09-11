#!/usr/bin/env python3
"""Run K=3-8 K-means selection on fixed-coordinate fused URs for CUBE Fig. 5B."""

import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score


FIXED_FUSION_ROOT = Path("./result/02_UMAP/data/fused_ur_fixed_coord")
FIG5A_UMAP_CSV = Path("./result/02_UMAP/result/fig5a_fixed_coord/fig5a_fixed_coord_umap_coordinates.csv")
OUT_DIR = Path("./result/03_spatial_cluster/result/fig5b_cluster_selection")

SPLITS = ("train", "val", "test")
K_VALUES = range(3, 9)
PCA_COMPONENTS = 50
RANDOM_STATE = 2026
KMEANS_N_INIT = 50
STABILITY_RUNS = 10
STABILITY_N_INIT = 10
FIGURE_DPI = 400


def parse_sample_name(sample_name):
    match = re.fullmatch(r"(.+)_patch_(\d+)_(\d+)", sample_name)
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_features():
    features, rows = [], []
    for split in SPLITS:
        paths = sorted((FIXED_FUSION_ROOT / split).glob("*.npy"))
        print(f"{split}: {len(paths)} fixed-coordinate fused URs")
        for path in paths:
            feature = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
            mega_id, patch_i, patch_j = parse_sample_name(path.stem)
            features.append(feature)
            rows.append({
                "split": split, "sample_name": path.stem, "mega_id": mega_id,
                "patch_i": patch_i, "patch_j": patch_j
            })
    return np.stack(features), pd.DataFrame(rows)


def load_umap(metadata):
    table = pd.read_csv(FIG5A_UMAP_CSV)
    columns = ["split", "sample_name", "pca50_euclidean_UMAP1", "pca50_euclidean_UMAP2"]
    table = table[columns].copy()
    return metadata.merge(table, on=["split", "sample_name"], how="left", validate="one_to_one")


def stability_score(x, k):
    labels = []
    for run in range(STABILITY_RUNS):
        model = KMeans(n_clusters=k, n_init=STABILITY_N_INIT, random_state=RANDOM_STATE + run)
        labels.append(model.fit_predict(x))
    scores = [adjusted_rand_score(labels[i], labels[j]) for i, j in combinations(range(STABILITY_RUNS), 2)]
    return float(np.mean(scores)), float(np.min(scores))


def run_clustering(x, metadata):
    assignments = metadata.copy()
    metric_rows, size_rows = [], []

    for k in K_VALUES:
        print(f"Clustering K={k}...")
        model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(x)
        assignments[f"k{k}"] = labels + 1

        counts = np.bincount(labels, minlength=k)
        stability_mean, stability_min = stability_score(x, k)
        metric_rows.append({
            "k": k,
            "silhouette": float(silhouette_score(x, labels, metric="euclidean")),
            "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
            "davies_bouldin": float(davies_bouldin_score(x, labels)),
            "inertia": float(model.inertia_),
            "stability_ari_mean": stability_mean,
            "stability_ari_min": stability_min,
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "min_cluster_fraction": float(counts.min() / len(labels))
        })
        for cluster_id, count in enumerate(counts, start=1):
            size_rows.append({"k": k, "cluster": cluster_id, "n": int(count), "fraction": float(count / len(labels))})

    return assignments, pd.DataFrame(metric_rows), pd.DataFrame(size_rows)


def save_umap_comparison(assignments):
    x = assignments["pca50_euclidean_UMAP1"].to_numpy()
    y = assignments["pca50_euclidean_UMAP2"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    axes = axes.ravel()

    for ax, k in zip(axes, K_VALUES):
        labels = assignments[f"k{k}"].to_numpy()
        scatter = ax.scatter(x, y, c=labels, cmap="tab10", s=6, alpha=0.80, linewidths=0, rasterized=True)
        ax.set_title(f"K={k}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        handles, _ = scatter.legend_elements()
        ax.legend(handles, [f"C{i}" for i in range(1, k + 1)], loc="best", frameon=False, fontsize=7, ncol=2)

    fig.suptitle("CUBE fixed-coordinate fused-UR K-means comparison", fontsize=15, fontweight="bold")
    fig.savefig(OUT_DIR / "k3_to_k8_umap_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k3_to_k8_umap_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(metrics):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", label="Silhouette")
    ax.plot(metrics["k"], metrics["stability_ari_mean"], marker="o", label="Mean stability ARI")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_xticks(list(K_VALUES))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_selection_primary_metrics.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k_selection_primary_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features, metadata = load_features()
    print(f"Loaded fused UR matrix: {features.shape}")

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    x = pca.fit_transform(features)
    print(f"PCA50 explained variance: {pca.explained_variance_ratio_.sum():.6f}")

    metadata = load_umap(metadata)
    assignments, metrics, cluster_sizes = run_clustering(x, metadata)

    assignments.to_csv(OUT_DIR / "k3_to_k8_cluster_assignments.csv", index=False)
    metrics.to_csv(OUT_DIR / "k3_to_k8_cluster_metrics.csv", index=False, float_format="%.8g")
    cluster_sizes.to_csv(OUT_DIR / "k3_to_k8_cluster_sizes.csv", index=False, float_format="%.8g")

    pd.DataFrame({
        "PC": np.arange(1, PCA_COMPONENTS + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(OUT_DIR / "pca50_explained_variance.csv", index=False, float_format="%.10g")

    save_umap_comparison(assignments)
    save_metric_plot(metrics)

    report = {
        "analysis": "CUBE Fig. 5B fixed-coordinate fused-UR K-means selection",
        "n_samples": int(len(metadata)),
        "feature_dimension": int(features.shape[1]),
        "representation": "fixed-coordinate fused UR",
        "clustering_input": "PCA50 without StandardScaler",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "k_values": list(K_VALUES),
        "kmeans_n_init": KMEANS_N_INIT,
        "stability_runs": STABILITY_RUNS,
        "stability_n_init": STABILITY_N_INIT,
        "random_state": RANDOM_STATE,
        "selection_note": "Compare silhouette, Davies-Bouldin, Calinski-Harabasz, cluster balance, stability ARI, and UMAP structure before fixing K."
    }
    with (OUT_DIR / "cluster_selection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("K=3-8 CLUSTERING COMPLETE")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
