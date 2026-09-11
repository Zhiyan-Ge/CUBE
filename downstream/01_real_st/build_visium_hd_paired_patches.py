from pathlib import Path
import json

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from scipy import sparse
from tqdm import tqdm


# Edit these paths independently when files move.
BTF_PATH = Path("./data/paired_data/he_st/raw_data/sample01/Visium_HD_Human_Colon_Cancer_binned_outputs/binned_outputs/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf")
H5_PATH = Path("./data/paired_data/he_st/raw_data/sample01/Visium_HD_Human_Colon_Cancer_binned_outputs/binned_outputs/square_016um/filtered_feature_bc_matrix.h5")
POSITIONS_PATH = Path("./data/paired_data/he_st/raw_data/sample01/Visium_HD_Human_Colon_Cancer_binned_outputs/binned_outputs/square_016um/spatial/tissue_positions.parquet")
GENE_INFO_CSV = Path("./data_preprocessing/DeepSpot_model/info_highly_variable_genes.csv")
OUT_DIR = Path("./result/01_real_st/data/HD/visium_hd_16um_paired")

GRID_SIZE = 16
BIN_SIZE_UM = 16.0
OUTPUT_SIZE = 1024
TARGET_GENE_COUNT = 256
CP10K_SCALE = 10000.0
MIN_FILTERED_BINS = 1
READ_MARGIN_PX = 4
MAX_PATCHES = None


def decode_strings(values):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def read_10x_h5(path):
    with h5py.File(path, "r") as f:
        group = f["matrix"]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group["shape"][:]),
        ).T.tocsr()
        barcodes = decode_strings(group["barcodes"][:])
        names = decode_strings(group["features"]["name"][:])
        if "feature_type" in group["features"]:
            feature_types = decode_strings(group["features"]["feature_type"][:])
            keep = feature_types == "Gene Expression"
            matrix = matrix[:, keep].tocsr()
            names = names[keep]
    return matrix, barcodes, names


def load_cube_genes(path):
    genes = pd.read_csv(path)
    predicted = genes["isPredicted"] if genes["isPredicted"].dtype == bool else genes["isPredicted"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    selected = genes.loc[predicted].copy()
    selected["deepspot_prediction_index"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.sort_values("highly_variable_rank", ascending=True).head(TARGET_GENE_COUNT).reset_index(drop=True)
    selected.insert(0, "model_channel", np.arange(len(selected), dtype=np.int64))
    return selected[["model_channel", "gene_name", "deepspot_prediction_index"]]


def fit_grid_affine(positions):
    design = np.column_stack([
        np.ones(len(positions), dtype=np.float64),
        positions["array_col"].to_numpy(dtype=np.float64),
        positions["array_row"].to_numpy(dtype=np.float64),
    ])
    coef_x = np.linalg.lstsq(design, positions["pxl_col_in_fullres"].to_numpy(dtype=np.float64), rcond=None)[0]
    coef_y = np.linalg.lstsq(design, positions["pxl_row_in_fullres"].to_numpy(dtype=np.float64), rcond=None)[0]
    return coef_x, coef_y


def grid_xy(array_row, array_col, coef_x, coef_y):
    return np.asarray([
        coef_x[0] + coef_x[1] * array_col + coef_x[2] * array_row,
        coef_y[0] + coef_y[1] * array_col + coef_y[2] * array_row,
    ], dtype=np.float64)


def block_geometry(r0, c0, coef_x, coef_y):
    # Output row 0 is the top of the image. Here array_row increases upward,
    # so array rows are reversed when mapped into the output patch.
    corners = np.stack([
        grid_xy(r0 + 15.5, c0 - 0.5, coef_x, coef_y),
        grid_xy(r0 + 15.5, c0 + 15.5, coef_x, coef_y),
        grid_xy(r0 - 0.5, c0 + 15.5, coef_x, coef_y),
        grid_xy(r0 - 0.5, c0 - 0.5, coef_x, coef_y),
    ])
    source_centers = np.stack([
        grid_xy(r0 + 15, c0, coef_x, coef_y),
        grid_xy(r0 + 15, c0 + 15, coef_x, coef_y),
        grid_xy(r0, c0, coef_x, coef_y),
    ])
    return corners, source_centers


def block_inside_image(corners, width, height):
    x, y = corners[:, 0], corners[:, 1]
    return bool(
        x.min() >= READ_MARGIN_PX and y.min() >= READ_MARGIN_PX
        and x.max() < width - READ_MARGIN_PX and y.max() < height - READ_MARGIN_PX
    )


def extract_aligned_patch(slide, corners, source_centers):
    left = int(np.floor(corners[:, 0].min())) - READ_MARGIN_PX
    top = int(np.floor(corners[:, 1].min())) - READ_MARGIN_PX
    right = int(np.ceil(corners[:, 0].max())) + READ_MARGIN_PX
    bottom = int(np.ceil(corners[:, 1].max())) + READ_MARGIN_PX
    region = np.asarray(slide.read_region((left, top), 0, (right - left, bottom - top)).convert("RGB"))

    source_local = source_centers - np.asarray([left, top], dtype=np.float64)
    cell_size = OUTPUT_SIZE / GRID_SIZE
    target_centers = np.asarray([
        [cell_size / 2, cell_size / 2],
        [OUTPUT_SIZE - cell_size / 2, cell_size / 2],
        [cell_size / 2, OUTPUT_SIZE - cell_size / 2],
    ], dtype=np.float32)
    transform = cv2.getAffineTransform(source_local.astype(np.float32), target_centers)
    return cv2.warpAffine(
        region, transform, (OUTPUT_SIZE, OUTPUT_SIZE), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def build_st_target(block, r0, c0, gex, total_counts, model_var_indices, gene_mask):
    st_raw = np.zeros((GRID_SIZE, GRID_SIZE, TARGET_GENE_COUNT), dtype=np.float32)
    st_log1p = np.zeros_like(st_raw)
    bin_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    total_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    array_rows = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int16)
    array_cols = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int16)
    x_fullres = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
    y_fullres = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
    barcode_grid = np.full((GRID_SIZE, GRID_SIZE), "", dtype="<U32")

    matrix_rows = block["matrix_index"].to_numpy(dtype=np.int64)
    matched_channels = np.flatnonzero(gene_mask)
    matched_counts = gex[matrix_rows][:, model_var_indices[gene_mask]].toarray().astype(np.float32)
    libraries = total_counts[matrix_rows]
    matched_log1p = np.zeros_like(matched_counts)
    positive = libraries > 0
    matched_log1p[positive] = np.log1p(matched_counts[positive] / libraries[positive, None] * CP10K_SCALE)

    bin_records = []
    for k, row in enumerate(block.itertuples(index=False)):
        output_row = 15 - (int(row.array_row) - r0)
        output_col = int(row.array_col) - c0
        st_raw[output_row, output_col, matched_channels] = matched_counts[k]
        st_log1p[output_row, output_col, matched_channels] = matched_log1p[k]
        bin_mask[output_row, output_col] = True
        total_grid[output_row, output_col] = libraries[k]
        array_rows[output_row, output_col] = int(row.array_row)
        array_cols[output_row, output_col] = int(row.array_col)
        x_fullres[output_row, output_col] = float(row.pxl_col_in_fullres)
        y_fullres[output_row, output_col] = float(row.pxl_row_in_fullres)
        barcode_grid[output_row, output_col] = str(row.barcode)
        bin_records.append({
            "output_row": output_row,
            "output_col": output_col,
            "barcode": str(row.barcode),
            "array_row": int(row.array_row),
            "array_col": int(row.array_col),
            "x_fullres": float(row.pxl_col_in_fullres),
            "y_fullres": float(row.pxl_row_in_fullres),
            "total_counts": float(libraries[k]),
        })

    target = {
        "st_log1p_cp10k": st_log1p,
        "st_raw_counts": st_raw,
        "bin_mask": bin_mask,
        "gene_mask": gene_mask,
        "total_counts": total_grid,
        "array_row": array_rows,
        "array_col": array_cols,
        "x_fullres": x_fullres,
        "y_fullres": y_fullres,
        "barcodes": barcode_grid,
    }
    return target, bin_records


def main():
    patch_dir = OUT_DIR / "raw_patches_1024"
    target_dir = OUT_DIR / "st_targets"
    patch_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    gex, barcodes, var_names = read_10x_h5(H5_PATH)
    total_counts = np.asarray(gex.sum(axis=1)).ravel().astype(np.float32)
    positions = pd.read_parquet(POSITIONS_PATH)
    if "barcode" not in positions.columns:
        positions = positions.reset_index().rename(columns={positions.index.name or "index": "barcode"})
    positions["barcode"] = positions["barcode"].astype(str)
    positions[["array_row", "array_col"]] = positions[["array_row", "array_col"]].astype(int)

    barcode_to_index = {barcode: i for i, barcode in enumerate(barcodes)}
    filtered = positions[positions["barcode"].isin(barcode_to_index)].copy()
    filtered["matrix_index"] = filtered["barcode"].map(barcode_to_index).astype(np.int64)
    filtered["block_row"] = filtered["array_row"] // GRID_SIZE
    filtered["block_col"] = filtered["array_col"] // GRID_SIZE

    gene_mapping = load_cube_genes(GENE_INFO_CSV)
    var_lookup = {name: i for i, name in enumerate(var_names)}
    model_var_indices = np.asarray([var_lookup.get(name, -1) for name in gene_mapping["gene_name"]], dtype=np.int64)
    gene_mask = model_var_indices >= 0
    gene_mapping["visium_hd_var_index"] = pd.array(np.where(gene_mask, model_var_indices, -1), dtype="Int64")
    gene_mapping.loc[~gene_mask, "visium_hd_var_index"] = pd.NA
    gene_mapping["in_visium_hd"] = gene_mask
    gene_mapping.to_csv(OUT_DIR / "gene_mapping.csv", index=False)

    coef_x, coef_y = fit_grid_affine(positions)
    col_vector = np.asarray([coef_x[1], coef_y[1]])
    row_vector = np.asarray([coef_x[2], coef_y[2]])
    mean_spacing = float((np.linalg.norm(col_vector) + np.linalg.norm(row_vector)) / 2)
    inferred_mpp = BIN_SIZE_UM / mean_spacing

    slide = openslide.OpenSlide(str(BTF_PATH))
    width, height = slide.dimensions
    block_records, bin_records = [], []
    dropped_boundary = 0

    groups = list(filtered.groupby(["block_row", "block_col"], sort=True))
    progress = tqdm(groups, desc="Build paired patches")
    for (block_row, block_col), block in progress:
        r0, c0 = int(block_row) * GRID_SIZE, int(block_col) * GRID_SIZE
        if r0 + 15 > positions["array_row"].max() or c0 + 15 > positions["array_col"].max():
            continue
        if len(block) < MIN_FILTERED_BINS:
            continue

        corners, source_centers = block_geometry(r0, c0, coef_x, coef_y)
        if not block_inside_image(corners, width, height):
            dropped_boundary += 1
            continue

        patch_id = f"block_r{int(block_row):03d}_c{int(block_col):03d}"
        patch_path = patch_dir / f"{patch_id}.png"
        target_path = target_dir / f"{patch_id}.npz"

        patch = extract_aligned_patch(slide, corners, source_centers)
        Image.fromarray(patch).save(patch_path)
        target, current_bin_records = build_st_target(
            block, r0, c0, gex, total_counts, model_var_indices, gene_mask,
        )
        np.savez_compressed(target_path, **target)

        for record in current_bin_records:
            record["patch_id"] = patch_id
        bin_records.extend(current_bin_records)
        block_records.append({
            "patch_id": patch_id,
            "block_row": int(block_row),
            "block_col": int(block_col),
            "array_row_min": r0,
            "array_row_max": r0 + 15,
            "array_col_min": c0,
            "array_col_max": c0 + 15,
            "filtered_bins": int(len(block)),
            "filtered_fraction": float(len(block) / (GRID_SIZE * GRID_SIZE)),
            "patch_path": str(patch_path.relative_to(OUT_DIR)),
            "st_target_path": str(target_path.relative_to(OUT_DIR)),
            "top_left_x": float(corners[0, 0]),
            "top_left_y": float(corners[0, 1]),
            "top_right_x": float(corners[1, 0]),
            "top_right_y": float(corners[1, 1]),
            "bottom_right_x": float(corners[2, 0]),
            "bottom_right_y": float(corners[2, 1]),
            "bottom_left_x": float(corners[3, 0]),
            "bottom_left_y": float(corners[3, 1]),
        })
        if MAX_PATCHES is not None and len(block_records) >= MAX_PATCHES:
            break

    slide.close()
    pd.DataFrame(block_records).to_csv(OUT_DIR / "patch_metadata.csv", index=False)
    pd.DataFrame(bin_records).to_csv(OUT_DIR / "bin_metadata.csv", index=False)

    report = {
        "btf_path": str(BTF_PATH),
        "h5_path": str(H5_PATH),
        "positions_path": str(POSITIONS_PATH),
        "gene_info_csv": str(GENE_INFO_CSV),
        "output_dir": str(OUT_DIR),
        "patches": len(block_records),
        "filtered_bins_in_h5": int(len(barcodes)),
        "filtered_bins_exported": int(len(bin_records)),
        "blocks_dropped_at_btf_boundary": dropped_boundary,
        "image_size": [width, height],
        "patch_physical_size_um": GRID_SIZE * BIN_SIZE_UM,
        "patch_output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
        "st_shape": [GRID_SIZE, GRID_SIZE, TARGET_GENE_COUNT],
        "bin_output_size_px": OUTPUT_SIZE // GRID_SIZE,
        "mean_grid_spacing_fullres_px": mean_spacing,
        "inferred_mpp_um_per_px": inferred_mpp,
        "matched_model_genes": int(gene_mask.sum()),
        "missing_model_genes": gene_mapping.loc[~gene_mask, "gene_name"].tolist(),
        "normalization": "log1p(count / all-GEX library size * 10000)",
        "orientation": "output rows top-to-bottom correspond to decreasing Visium HD array_row",
    }
    with (OUT_DIR / "build_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("VISIUM HD PAIRED PATCH BUILD COMPLETE")
    print("=" * 80)
    print(f"patches: {len(block_records)}")
    print(f"filtered bins exported: {len(bin_records)}/{len(barcodes)}")
    print(f"model genes matched: {gene_mask.sum()}/{TARGET_GENE_COUNT}")
    print(f"boundary blocks dropped: {dropped_boundary}")
    print(f"inferred MPP: {inferred_mpp:.8f} um/px")
    print(f"results: {OUT_DIR}")


if __name__ == "__main__":
    main()