#!/usr/bin/env python3
"""Run Fig. 5A UMAP analysis on fixed-coordinate CUBE fused representations."""

import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import umap
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


# Edit only this section if files move.
PKL_DIRS = {
    "train": Path("./data/train_data/final_data/train/pkl"),
    "val": Path("./data/train_data/final_data/val/pkl"),
    "test": Path("./data/train_data/final_data/test/pkl"),
}
FIXED_FUSION_ROOT = Path(
    "./result/02_UMAP/data/fused_ur_fixed_coord"
)
FUSED_DIRS = {split: FIXED_FUSION_ROOT / split for split in ("train", "val", "test")}
TISSUE_FRACTION_CSV = Path(
    "./result/02_UMAP/data/tissue_fraction_per_patch.csv"
)
OUT_DIR = Path(
    "./result/02_UMAP/result/fig5a_fixed_coord"
)


SPLITS = ("train", "val", "test")
TARGETS = ("DAPI", "CD3", "panCK", "tissue_fraction", "coord_x", "coord_y")
EXPECTED_DIM = 512
PCA_COMPONENTS = 50
K_VALUES = (10, 30)
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.25
RANDOM_STATE = 2026
COLOR_LOW_QUANTILE = 0.01
COLOR_HIGH_QUANTILE = 0.99
POINT_SIZE = 6.0
POINT_ALPHA = 0.78
FIGURE_DPI = 400
PRIMARY_METHOD = "pca50_euclidean"


def normalize_sample_name(value):
    name = Path(str(value).strip()).name
    lowered = name.lower()
    for suffix in (".pkl", ".npy", ".tif", ".tiff"):
        if lowered.endswith(suffix):
            return name[:-len(suffix)]
    return name


def stems(path, suffix):
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    suffix = suffix.lower()
    return {file.stem for file in path.iterdir() if file.is_file() and file.suffix.lower() == suffix}


def mismatch_lines(reference, observed, label, limit=10):
    missing = sorted(reference - observed)
    extra = sorted(observed - reference)
    lines = [f"  {label}: total={len(observed)}, missing={len(missing)}, extra={len(extra)}"]
    if missing:
        lines.append(f"    missing examples: {missing[:limit]}")
    if extra:
        lines.append(f"    extra examples: {extra[:limit]}")
    return lines


def load_export_manifest():
    path = FIXED_FUSION_ROOT / "fixed_coord_export_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fixed-coordinate export manifest not found: {path}")
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Fixed-coordinate export is not completed: status={manifest.get('status')}")
    if int(manifest.get("expected_dimension", -1)) != EXPECTED_DIM:
        raise RuntimeError(f"Manifest dimension mismatch: {manifest.get('expected_dimension')}")
    final_counts = manifest.get("final_counts", {})
    expected_counts = {"train": 3717, "val": 630, "test": 945}
    if {key: int(final_counts.get(key, -1)) for key in SPLITS} != expected_counts:
        raise RuntimeError(f"Unexpected manifest counts: {final_counts}")
    fixed_coord = np.asarray(manifest.get("coordinate", {}).get("fixed_coordinate", []), dtype=np.float64)
    if fixed_coord.shape != (2,) or not np.isfinite(fixed_coord).all():
        raise RuntimeError(f"Invalid fixed coordinate in manifest: {fixed_coord}")
    return manifest


def load_tissue_table():
    if not TISSUE_FRACTION_CSV.is_file():
        raise FileNotFoundError(f"Tissue-fraction CSV not found: {TISSUE_FRACTION_CSV}")
    table = pd.read_csv(TISSUE_FRACTION_CSV)
    required = {"split", "sample_name", "tissue_fraction"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"Tissue CSV is missing columns: {missing}")
    table = table.copy()
    table["split"] = table["split"].astype(str).str.strip().str.lower()
    table["sample_name"] = table["sample_name"].map(normalize_sample_name)
    table["tissue_fraction"] = pd.to_numeric(table["tissue_fraction"], errors="raise")
    duplicated = table.duplicated(["split", "sample_name"], keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, ["split", "sample_name"]].head(10)
        raise RuntimeError(f"Duplicate tissue CSV keys:\n{examples.to_string(index=False)}")
    values = table["tissue_fraction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError("tissue_fraction must contain finite values in [0, 1].")
    return table.set_index(["split", "sample_name"], drop=False).sort_index()


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def load_pkl_metadata(path):
    with path.open("rb") as file:
        payload = pickle.load(file)
    missing = [key for key in ("concept_scores", "normalized_mega_coord") if key not in payload]
    if missing:
        raise RuntimeError(f"{path} is missing keys: {missing}")
    concepts = as_numpy(payload["concept_scores"]).astype(np.float64, copy=False).reshape(-1)
    coord = as_numpy(payload["normalized_mega_coord"]).astype(np.float64, copy=False).reshape(-1)
    if concepts.shape != (3,) or coord.shape != (2,):
        raise RuntimeError(f"Unexpected PKL shapes in {path}: concepts={concepts.shape}, coord={coord.shape}")
    if not np.isfinite(concepts).all() or not np.isfinite(coord).all():
        raise RuntimeError(f"Non-finite PKL metadata: {path}")
    return concepts, coord


def parse_patch_name(sample_name):
    match = re.fullmatch(r"(.+)_patch_(\d+)_(\d+)", sample_name)
    if match is None:
        raise RuntimeError(f"Cannot parse HEMIT patch name: {sample_name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_feature(path):
    feature = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if feature.shape != (EXPECTED_DIM,) or not np.isfinite(feature).all():
        raise RuntimeError(f"Invalid fixed-coordinate fusion: {path}, shape={feature.shape}")
    return feature


def load_all_samples():
    tissue = load_tissue_table()
    features, records = [], []
    for split in SPLITS:
        pkl_names = stems(PKL_DIRS[split], ".pkl")
        fused_names = stems(FUSED_DIRS[split], ".npy")
        tissue_names = set(tissue.loc[tissue["split"] == split, "sample_name"].tolist())
        if pkl_names != fused_names or pkl_names != tissue_names:
            lines = [f"Sample mismatch in {split}; PKL is the reference (n={len(pkl_names)})."]
            lines.extend(mismatch_lines(pkl_names, fused_names, "fixed fusion"))
            lines.extend(mismatch_lines(pkl_names, tissue_names, "tissue CSV"))
            raise RuntimeError("\n".join(lines))
        print(f"Loading fixed-coordinate fusion {split}: {len(pkl_names)} patches")
        for sample_name in sorted(pkl_names):
            feature = load_feature(FUSED_DIRS[split] / f"{sample_name}.npy")
            concepts, coord = load_pkl_metadata(PKL_DIRS[split] / f"{sample_name}.pkl")
            tissue_row = tissue.loc[(split, sample_name), :]
            mega_id, patch_i, patch_j = parse_patch_name(sample_name)
            records.append({
                "split": split, "sample_name": sample_name, "mega_id": mega_id,
                "mega_label": f"{split}|{mega_id}", "patch_i": patch_i, "patch_j": patch_j,
                "DAPI": float(concepts[0]), "CD3": float(concepts[1]), "panCK": float(concepts[2]),
                "tissue_fraction": float(tissue_row["tissue_fraction"]),
                "coord_x": float(coord[0]), "coord_y": float(coord[1]),
            })
            features.append(feature)
    matrix = np.stack(features).astype(np.float32, copy=False)
    metadata = pd.DataFrame.from_records(records)
    print(f"Loaded all samples: fixed-coordinate feature matrix={matrix.shape}")
    return matrix, metadata


def build_methods(feature_matrix):
    n_components = min(PCA_COMPONENTS, feature_matrix.shape[0] - 1, feature_matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
    pca_scores = pca.fit_transform(feature_matrix).astype(np.float32, copy=False)
    methods = [
        {
            "name": "raw_cosine", "title": "Fixed-coordinate fusion: raw + cosine",
            "matrix": feature_matrix, "metric": "cosine",
        },
        {
            "name": "pca50_euclidean", "title": "Fixed-coordinate fusion: PCA-50 + Euclidean",
            "matrix": pca_scores, "metric": "euclidean",
        },
    ]
    return methods, pca


def chance_purity(labels):
    _, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def safe_pearson(observed, predicted):
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(pearsonr(observed, predicted).statistic)


def safe_spearman(observed, predicted):
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def neighbor_sets(matrix, metric, mega_labels):
    max_k = max(K_VALUES)
    _, group_sizes = np.unique(mega_labels, return_counts=True)
    search_without_self = min(len(matrix) - 1, int(group_sizes.max()) + max_k + 5)
    model = NearestNeighbors(
        n_neighbors=search_without_self + 1, metric=metric, algorithm="brute", n_jobs=-1
    ).fit(matrix)
    raw = model.kneighbors(matrix, return_distance=False)
    cleaned = []
    for index, row in enumerate(raw):
        row = row[row != index][:search_without_self]
        if len(row) < max_k:
            raise RuntimeError(f"Insufficient neighbors for sample {index}.")
        cleaned.append(row)
    candidates = np.stack(cleaned)
    neighbors = candidates[:, :max_k]
    cross_mega = []
    for index, row in enumerate(candidates):
        selected = row[mega_labels[row] != mega_labels[index]][:max_k]
        if len(selected) < max_k:
            raise RuntimeError(f"Insufficient cross-mega neighbors for sample {index}.")
        cross_mega.append(selected)
    return neighbors, np.stack(cross_mega)


def evaluate_method(method, metadata):
    print(f"Auditing {method['name']}: shape={method['matrix'].shape}, metric={method['metric']}")
    mega = metadata["mega_label"].to_numpy(dtype=str)
    split = metadata["split"].to_numpy(dtype=str)
    neighbors, cross_mega = neighbor_sets(method["matrix"], method["metric"], mega)
    result = {
        "method": method["name"], "dimension": int(method["matrix"].shape[1]), "metric": method["metric"],
        "mega_chance_purity": chance_purity(mega), "split_chance_purity": chance_purity(split),
    }
    for k in K_VALUES:
        selected, selected_cross = neighbors[:, :k], cross_mega[:, :k]
        result[f"mega_purity_k{k}"] = float((mega[selected] == mega[:, None]).mean())
        result[f"split_purity_k{k}"] = float((split[selected] == split[:, None]).mean())
        result[f"cross_mega_split_purity_k{k}"] = float((split[selected_cross] == split[:, None]).mean())
    for target in TARGETS:
        observed = metadata[target].to_numpy(dtype=np.float64)
        std = float(np.std(observed))
        for scope, indices in (("all", neighbors), ("cross_mega", cross_mega)):
            predicted = observed[indices[:, :max(K_VALUES)]].mean(axis=1)
            result[f"{target}_{scope}_pearson"] = safe_pearson(observed, predicted)
            result[f"{target}_{scope}_spearman"] = safe_spearman(observed, predicted)
            result[f"{target}_{scope}_normalized_mae"] = (
                float(np.mean(np.abs(observed - predicted)) / std) if std > 0 else float("nan")
            )
    return result


def run_umap(method):
    print(
        f"Running UMAP {method['name']}: n_neighbors={UMAP_N_NEIGHBORS}, "
        f"min_dist={UMAP_MIN_DIST}, metric={method['metric']}, seed={RANDOM_STATE}"
    )
    reducer = UMAP(
        n_components=2, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric=method["metric"], init="spectral", random_state=RANDOM_STATE,
        transform_seed=RANDOM_STATE, n_jobs=1, low_memory=True, verbose=True,
    )
    embedding = reducer.fit_transform(method["matrix"]).astype(np.float32, copy=False)
    if embedding.shape != (len(method["matrix"]), 2) or not np.isfinite(embedding).all():
        raise RuntimeError(f"Invalid UMAP result for {method['name']}: {embedding.shape}")
    return embedding


def padded_limits(values, fraction=0.04):
    low, high = float(np.min(values)), float(np.max(values))
    padding = fraction * (high - low) if high > low else 1.0
    return low - padding, high + padding


def style_axis(axis, title, panel_label, x_limits, y_limits):
    axis.set_title(title, fontsize=12, pad=8)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        -0.06, 1.04, panel_label, transform=axis.transAxes, fontsize=14,
        fontweight="bold", va="bottom", ha="right",
    )


def scatter_continuous(fig, axis, embedding, values, cmap, colorbar_label):
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = np.quantile(values, [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE])
    if vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    clipped = np.clip(values, vmin, vmax)
    order = np.argsort(clipped, kind="stable")
    scatter = axis.scatter(
        embedding[order, 0], embedding[order, 1], s=POINT_SIZE, c=clipped[order],
        cmap=cmap, vmin=vmin, vmax=vmax, alpha=POINT_ALPHA,
        linewidths=0, edgecolors="none", rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8, length=2)
    return [float(vmin), float(vmax)]


def scatter_split(axis, embedding, metadata):
    colors = {"train": "#7A7A7A", "val": "#D55E00", "test": "#0072B2"}
    for split in SPLITS:
        mask = metadata["split"].to_numpy() == split
        axis.scatter(
            embedding[mask, 0], embedding[mask, 1], s=POINT_SIZE, c=colors[split],
            alpha=0.70, linewidths=0, edgecolors="none",
            label=f"{split} (n={int(mask.sum())})", rasterized=True,
        )
    axis.legend(loc="best", frameon=False, fontsize=8, markerscale=2.2, handletextpad=0.4)


def save_six_panel(method, embedding, metadata):
    x_limits, y_limits = padded_limits(embedding[:, 0]), padded_limits(embedding[:, 1])
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.6), constrained_layout=True)
    axes = axes.ravel()
    axes[0].scatter(
        embedding[:, 0], embedding[:, 1], s=POINT_SIZE, c="#6F6F6F",
        alpha=0.64, linewidths=0, edgecolors="none", rasterized=True,
    )
    style_axis(axes[0], "Fixed-coordinate fused-UR geometry", "A", x_limits, y_limits)
    display_ranges = {}
    panels = (
        (1, "DAPI", "Blues", "B"), (2, "CD3", "Greens", "C"),
        (3, "panCK", "Reds", "D"), (4, "tissue_fraction", "viridis", "E"),
    )
    for axis_index, column, cmap, label in panels:
        display_ranges[column] = scatter_continuous(
            fig, axes[axis_index], embedding, metadata[column].to_numpy(), cmap, column.replace("_", " ")
        )
        style_axis(
            axes[axis_index], "Tissue fraction" if column == "tissue_fraction" else column,
            label, x_limits, y_limits,
        )
    scatter_split(axes[5], embedding, metadata)
    style_axis(axes[5], "Data split (patient-disjoint cohorts)", "F", x_limits, y_limits)
    fig.suptitle(f"CUBE fixed-coordinate fused-UR UMAP — {method['title']}", fontsize=15, fontweight="bold")
    stem = f"fig5a_fixed_coord_{method['name']}_umap"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return display_ranges


def category_colors(metadata):
    labels = metadata["mega_label"].astype("category")
    n_categories = len(labels.cat.categories)
    rng = np.random.default_rng(RANDOM_STATE)
    palette = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, n_categories))[rng.permutation(n_categories)]
    return palette[labels.cat.codes.to_numpy()], n_categories


def simple_axis(axis, embedding):
    x_limits, y_limits = padded_limits(embedding[:, 0]), padded_limits(embedding[:, 1])
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_method_comparison(methods, embeddings, metadata):
    colors, n_mega = category_colors(metadata)
    split_colors = {"train": "#7A7A7A", "val": "#D55E00", "test": "#0072B2"}
    fig, axes = plt.subplots(2, len(methods), figsize=(8.4, 8.0), constrained_layout=True)
    for column, method in enumerate(methods):
        embedding = embeddings[method["name"]]
        axes[0, column].scatter(
            embedding[:, 0], embedding[:, 1], s=5, c=colors,
            alpha=0.80, linewidths=0, rasterized=True,
        )
        axes[0, column].set_title(method["title"], fontsize=11, pad=8)
        for split in SPLITS:
            mask = metadata["split"].to_numpy() == split
            axes[1, column].scatter(
                embedding[mask, 0], embedding[mask, 1], s=5, c=split_colors[split],
                alpha=0.72, linewidths=0, label=split, rasterized=True,
            )
        simple_axis(axes[0, column], embedding)
        simple_axis(axes[1, column], embedding)
    axes[0, 0].set_ylabel(f"Parent region\n({n_mega} categorical colors)", fontsize=11, labelpad=12)
    axes[1, 0].set_ylabel("Data split", fontsize=11, labelpad=12)
    axes[1, -1].legend(loc="best", frameon=False, fontsize=8, markerscale=2)
    fig.suptitle("Fixed-coordinate fusion: UMAP method sensitivity", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "fig5a_fixed_coord_method_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5a_fixed_coord_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def save_original_coordinate_diagnostic(embedding, metadata):
    x_limits, y_limits = padded_limits(embedding[:, 0]), padded_limits(embedding[:, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for axis, column, label in zip(axes, ("coord_x", "coord_y"), ("A", "B")):
        scatter_continuous(fig, axis, embedding, metadata[column].to_numpy(), "coolwarm", f"original {column}")
        style_axis(axis, f"Original {column}", label, x_limits, y_limits)
    fig.suptitle("Original-coordinate association after fixed-coordinate encoding", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "fig5a_fixed_coord_original_coordinate_diagnostic.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5a_fixed_coord_original_coordinate_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def save_outputs(methods, embeddings, metadata, metrics, pca, manifest, display_ranges):
    coordinates = metadata.copy()
    for method in methods:
        embedding = embeddings[method["name"]]
        coordinates[f"{method['name']}_UMAP1"] = embedding[:, 0]
        coordinates[f"{method['name']}_UMAP2"] = embedding[:, 1]
    coordinates.to_csv(OUT_DIR / "fig5a_fixed_coord_umap_coordinates.csv", index=False, float_format="%.8g")
    metrics.to_csv(OUT_DIR / "fig5a_fixed_coord_geometry_metrics.csv", index=False, float_format="%.8g")

    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    pd.DataFrame({
        "PC": np.arange(1, len(explained) + 1),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance_ratio": np.cumsum(explained),
    }).to_csv(OUT_DIR / "fig5a_fixed_coord_pca_explained_variance.csv", index=False, float_format="%.10g")

    report = {
        "analysis": "CUBE Fig. 5A fixed-coordinate UMAP",
        "n_patches": int(len(metadata)),
        "split_counts": metadata["split"].value_counts().reindex(SPLITS).astype(int).to_dict(),
        "fixed_coordinate_export": manifest,
        "primary_method": PRIMARY_METHOD,
        "methods": {
            method["name"]: {
                "input_dimension": int(method["matrix"].shape[1]),
                "metric": method["metric"],
                "umap_n_neighbors": UMAP_N_NEIGHBORS,
                "umap_min_dist": UMAP_MIN_DIST,
                "random_state": RANDOM_STATE,
            }
            for method in methods
        },
        "pca": {
            "standard_scaler_used": False,
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "color_display": {
            "quantile_clip": [COLOR_LOW_QUANTILE, COLOR_HIGH_QUANTILE],
            "ranges": display_ranges,
            "raw_values_preserved_in_csv": True,
        },
        "interpretation_note": (
            "All patches were encoded with the same global coordinate. Original coord_x/coord_y are retained only "
            "as diagnostic metadata and were not supplied to model.encode for these cached representations."
        ),
        "software_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "umap_learn": umap.__version__,
        },
    }
    with (OUT_DIR / "fig5a_fixed_coord_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def print_summary(metrics):
    columns = [
        "method", "mega_purity_k10", "mega_purity_k30", "split_purity_k30",
        "cross_mega_split_purity_k30", "DAPI_cross_mega_spearman",
        "CD3_cross_mega_spearman", "panCK_cross_mega_spearman",
        "tissue_fraction_cross_mega_spearman", "coord_x_cross_mega_spearman",
        "coord_y_cross_mega_spearman",
    ]
    print("\nFIXED-COORDINATE GEOMETRY SUMMARY")
    print("=" * 36)
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_export_manifest()
    feature_matrix, metadata = load_all_samples()
    methods, pca = build_methods(feature_matrix)
    metrics = pd.DataFrame([evaluate_method(method, metadata) for method in methods])
    embeddings = {method["name"]: run_umap(method) for method in methods}

    display_ranges = {}
    for method in methods:
        display_ranges[method["name"]] = save_six_panel(method, embeddings[method["name"]], metadata)
    save_method_comparison(methods, embeddings, metadata)
    save_original_coordinate_diagnostic(embeddings[PRIMARY_METHOD], metadata)
    save_outputs(methods, embeddings, metadata, metrics, pca, manifest, display_ranges)
    print_summary(metrics)
    print(f"\nFixed-coordinate UMAP finished. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
