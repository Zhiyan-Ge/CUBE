#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUBE pseudo-ST preprocessing with DeepSpot.

This module predicts a spatial pseudo-ST expression matrix from an H&E patch
using a pretrained DeepSpot model.

Inputs:
    1. An H&E image path, typically a 1024x1024 patch.
    2. Alternatively, an H&E array with shape [3, 1024, 1024].

Output:
    ST_matrix: np.ndarray with shape [16, 16, 256]
        - 16 x 16: fixed CUBE spatial grid.
        - 256: selected highly variable genes from the DeepSpot prediction.

The module is intentionally limited to H&E -> pseudo-ST inference. It does not
handle the main image-loading workflow, pickle packaging, or ST Z-score
normalization.
"""

import os
import sys
import json
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import yaml
import torch
import numpy as np
import pandas as pd


@dataclass
class DeepSpotConfig:
    """
    Configuration for DeepSpot inference.

    Required resources:
        1. DeepSpot repository.
        2. Pretrained DeepSpot expression model weights.
        3. DeepSpot hyperparameter YAML file.
        4. DeepSpot gene information CSV file.
        5. Pathology foundation model weights when required, e.g. UNI.
    """
    deepspot_repo_path: str
    model_weights_path: str
    model_hparam_path: str
    gene_info_csv_path: str
    morphology_model_path: Optional[str] = None

    device: str = "cuda"

    # Fixed CUBE output settings
    grid_size: int = 16
    target_gene_count: int = 256

    # DeepSpot spot settings
    # If None, infer automatically from image size and grid_size.
    spot_diameter_px: Optional[int] = None
    spot_distance_px: Optional[int] = None

    # DeepSpot uses 9 mini-tiles by default.
    n_mini_tiles: int = 9

    # Neighborhood radius used by DeepSpot; 1 uses the immediate neighbors.
    neighbor_radius: int = 1

    # Whether to filter near-white background spots.
    # CUBE requires a fixed [16, 16, 256] grid, so the default is False.
    # If True, filtered spots are filled with zeros in the original grid.
    filter_white: bool = False
    white_cutoff: float = 200.0

    # Output post-processing
    clip_negative: bool = False
    dtype: str = "float32"


class DeepSpotPseudoSTGenerator:
    """
    Lightweight DeepSpot wrapper used by CUBE.

    Example:
        cfg = DeepSpotConfig(...)
        generator = DeepSpotPseudoSTGenerator(cfg)
        st = generator.predict_from_image_path("/path/to/he_patch.tif")
        # st.shape == [16, 16, 256]
    """

    def __init__(self, config: DeepSpotConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu"
        )

        self._validate_paths()
        self._prepare_deepspot_import()
        self._import_deepspot_modules()

        self.model_hparam = self._load_yaml(config.model_hparam_path)
        self.image_feature_model = self.model_hparam.get("image_feature_model", None)

        if self.image_feature_model is None:
            raise KeyError(
                f"image_feature_model is missing from hparam YAML: {config.model_hparam_path}"
            )

        self.selected_gene_indices, self.selected_gene_names = self._load_selected_gene_indices()

        self.model_expression = self._load_expression_model()
        self.morphology_model, self.preprocess, self.feature_dim = self._load_morphology_model()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        required_paths = {
            "deepspot_repo_path": self.config.deepspot_repo_path,
            "model_weights_path": self.config.model_weights_path,
            "model_hparam_path": self.config.model_hparam_path,
            "gene_info_csv_path": self.config.gene_info_csv_path,
        }

        for name, path in required_paths.items():
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"{name} does not exist or is empty: {path}")

        # Some morphology models, such as Inception, do not require local weights.
        # UNI, Hoptimus0, Phikon, and related models usually require a local weight path.
        if self.config.morphology_model_path is not None:
            if not os.path.exists(self.config.morphology_model_path):
                raise FileNotFoundError(
                    f"morphology_model_path does not exist: {self.config.morphology_model_path}"
                )

    def _prepare_deepspot_import(self) -> None:
        """
        Add the DeepSpot repository to sys.path.

        This remains harmless if DeepSpot has already been installed with
        ``python setup.py install``.
        """
        repo_path = os.path.abspath(self.config.deepspot_repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _import_deepspot_modules(self) -> None:
        """
        Import DeepSpot functions lazily so importing this module does not
        immediately require the complete DeepSpot environment.
        """
        try:
            from deepspot.utils.utils_image import (
                predict_spot_spatial_transcriptomics_from_image_path,
                get_morphology_model_and_preprocess,
                crop_tile,
            )
        except Exception as e:
            raise ImportError(
                "Unable to import DeepSpot. Please confirm:\n"
                "1. deepspot_repo_path is correct;\n"
                "2. DeepSpot has been installed from its repository;\n"
                "3. pyvips, anndata, scanpy, squidpy, and related dependencies are installed.\n"
                f"Underlying error: {repr(e)}"
            )

        self.predict_spot_fn = predict_spot_spatial_transcriptomics_from_image_path
        self.get_morphology_model_and_preprocess = get_morphology_model_and_preprocess
        self.crop_tile = crop_tile

        try:
            import anndata as ad
            import pyvips
        except Exception as e:
            raise ImportError(
                "DeepSpot inference requires anndata and pyvips.\n"
                f"Underlying error: {repr(e)}"
            )

        self.ad = ad
        self.pyvips = pyvips

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_expression_model(self):
        """
        Load the pretrained DeepSpot expression prediction model.

        The official DeepSpot notebook loads the pretrained object directly
        with torch.load and then applies .to(device) and .eval().
        """
        try:
            model = torch.load(
                self.config.model_weights_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions without weights_only.
            model = torch.load(
                self.config.model_weights_path,
                map_location=self.device,
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_morphology_model(self):
        """
        Load the pathology foundation model used by DeepSpot.
        """
        model_name = self.image_feature_model

        models_requiring_path = {"uni", "hoptimus0", "phikon", "phikonv2"}
        if model_name in models_requiring_path and self.config.morphology_model_path is None:
            raise ValueError(
                f"image_feature_model={model_name} requires morphology_model_path. "
                "Provide the corresponding foundation model weight path in DeepSpotConfig."
            )

        morphology_model, preprocess, feature_dim = self.get_morphology_model_and_preprocess(
            model_name=model_name,
            device=self.device,
            model_path=self.config.morphology_model_path,
        )

        morphology_model.to(self.device)
        morphology_model.eval()
        return morphology_model, preprocess, feature_dim

    # ------------------------------------------------------------------
    # Gene selection
    # ------------------------------------------------------------------
    @staticmethod
    def _to_bool_series(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])

    def _load_selected_gene_indices(self) -> Tuple[np.ndarray, List[str]]:
        """
        Select the top target_gene_count highly variable genes from the
        DeepSpot gene information CSV.

        The CSV typically contains:
            - isPredicted
            - gene_name
            - highly_variable_rank

        The function filters predicted genes, sorts them by
        highly_variable_rank when available, selects the top genes, and
        returns their column indices in the DeepSpot prediction matrix.
        """
        genes = pd.read_csv(self.config.gene_info_csv_path)

        if "isPredicted" not in genes.columns:
            raise KeyError(
                f"isPredicted column is missing from gene info CSV: {self.config.gene_info_csv_path}"
            )

        predicted_mask = self._to_bool_series(genes["isPredicted"])
        pred_genes = genes.loc[predicted_mask].copy()
        pred_genes["_pred_col_idx"] = np.arange(len(pred_genes), dtype=np.int64)

        if len(pred_genes) < self.config.target_gene_count:
            raise ValueError(
                f"DeepSpot provides only {len(pred_genes)} predicted genes, "
                f"fewer than the {self.config.target_gene_count} required by CUBE"
            )

        if "highly_variable_rank" in pred_genes.columns:
            pred_genes = pred_genes.sort_values("highly_variable_rank", ascending=True)

        top_genes = pred_genes.head(self.config.target_gene_count)

        if "gene_name" in top_genes.columns:
            gene_names = top_genes["gene_name"].astype(str).tolist()
        else:
            gene_names = [f"gene_{i}" for i in range(self.config.target_gene_count)]

        indices = top_genes["_pred_col_idx"].to_numpy(dtype=np.int64)
        return indices, gene_names

    # ------------------------------------------------------------------
    # Grid and AnnData construction
    # ------------------------------------------------------------------
    def _infer_spot_geometry(self, image_path: str) -> Tuple[int, int, int, int]:
        """
        Infer spot_diameter and spot_distance from the image size and grid_size.

        For a 1024x1024 image with grid_size=16:
            cell_size = 64
            spot_diameter = 64
            spot_distance = 64
        """
        image = self.pyvips.Image.new_from_file(image_path, access="sequential")
        height = int(image.height)
        width = int(image.width)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid image size: height={height}, width={width}, path={image_path}"
            )

        cell_h = height / float(self.config.grid_size)
        cell_w = width / float(self.config.grid_size)

        inferred_diameter = int(round(min(cell_h, cell_w)))
        inferred_distance = int(round(min(cell_h, cell_w)))

        spot_diameter = self.config.spot_diameter_px or inferred_diameter
        spot_distance = self.config.spot_distance_px or inferred_distance

        return height, width, spot_diameter, spot_distance

    def _build_grid_adata(self, image_path: str):
        """
        Construct a 16x16 DeepSpot spot grid for one H&E patch.

        Returns:
            adata_full: AnnData containing all 256 spots.
            full_obs: Full observation table used to restore filtered spots.
            keep_mask: Boolean array with shape [256].
        """
        height, width, spot_diameter, _ = self._infer_spot_geometry(image_path)

        # In DeepSpot crop_tile, x_pixel/y_pixel behave like row/column coordinates:
        # x_pixel follows image.height and y_pixel follows image.width in the official notebook.
        coord = []
        idx = 0
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                x_center = int(round((i + 0.5) * height / self.config.grid_size))
                y_center = int(round((j + 0.5) * width / self.config.grid_size))

                # Clamp spot centers to valid image boundaries.
                half = spot_diameter // 2
                x_center = max(half, min(height - half, x_center))
                y_center = max(half, min(width - half, y_center))

                coord.append({
                    "barcode": str(idx),
                    "x_array": i,
                    "y_array": j,
                    "x_pixel": x_center,
                    "y_pixel": y_center,
                    "grid_i": i,
                    "grid_j": j,
                    "sampleID": "cube_patch",
                })
                idx += 1

        obs = pd.DataFrame(coord)
        obs.index = obs["barcode"].astype(str).values

        # Initialize empty counts; DeepSpot only requires aligned shape and var indices.
        n_pred_genes = self._get_n_predicted_genes()
        counts = np.empty((len(obs), n_pred_genes), dtype=np.float32)

        adata_full = self.ad.AnnData(counts)
        adata_full.obs = obs.copy()
        adata_full.var.index = self._get_all_predicted_gene_names()

        keep_mask = np.ones(len(obs), dtype=bool)

        if self.config.filter_white:
            image = self.pyvips.Image.new_from_file(image_path, access="sequential")
            is_white = []
            for _, row in obs.iterrows():
                tile = self.crop_tile(
                    image=image,
                    x_pixel=int(row.x_pixel),
                    y_pixel=int(row.y_pixel),
                    cell_diameter=spot_diameter,
                )
                white_value = float(np.mean(tile[:, :, :3]))
                is_white.append(white_value)

            obs["is_white"] = np.array(is_white, dtype=np.float32)
            keep_mask = obs["is_white"].values <= self.config.white_cutoff

            adata_full.obs = obs
            adata_query = adata_full[keep_mask, :].copy()
            return adata_query, obs.copy(), keep_mask

        return adata_full, obs.copy(), keep_mask

    def _get_n_predicted_genes(self) -> int:
        genes = pd.read_csv(self.config.gene_info_csv_path)
        pred_mask = self._to_bool_series(genes["isPredicted"])
        return int(pred_mask.sum())

    def _get_all_predicted_gene_names(self) -> List[str]:
        genes = pd.read_csv(self.config.gene_info_csv_path)
        pred_mask = self._to_bool_series(genes["isPredicted"])
        pred_genes = genes.loc[pred_mask].copy()

        if "gene_name" in pred_genes.columns:
            return pred_genes["gene_name"].astype(str).tolist()

        return [f"gene_{i}" for i in range(len(pred_genes))]

    # ------------------------------------------------------------------
    # Public inference interface
    # ------------------------------------------------------------------
    def predict_from_image_path(self, image_path: str) -> np.ndarray:
        """
        Generate the CUBE pseudo-ST matrix from an H&E image path.

        Args:
            image_path: H&E patch path, typically 1024x1024.

        Returns:
            st_matrix: np.ndarray with shape [16, 16, 256].
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"H&E image does not exist: {image_path}")

        height, width, spot_diameter, _ = self._infer_spot_geometry(image_path)

        adata_query, full_obs, keep_mask = self._build_grid_adata(image_path)

        if len(adata_query) == 0:
            return np.zeros(
                (
                    self.config.grid_size,
                    self.config.grid_size,
                    self.config.target_gene_count,
                ),
                dtype=np.float32,
            )

        counts_pred = self.predict_spot_fn(
            image_path,
            adata_query,
            spot_diameter,
            self.config.n_mini_tiles,
            self.preprocess,
            self.morphology_model,
            self.model_expression,
            self.device,
            super_resolution=False,
            neighbor_radius=self.config.neighbor_radius,
        )

        counts_pred = np.asarray(counts_pred, dtype=np.float32)

        # DeepSpot may return [G] for a single spot; standardize to [N, G].
        if counts_pred.ndim == 1:
            counts_pred = counts_pred[None, :]

        if counts_pred.ndim != 2:
            raise ValueError(f"Unexpected DeepSpot output shape: {counts_pred.shape}")

        if counts_pred.shape[1] < self.config.target_gene_count:
            raise ValueError(
                f"DeepSpot returned {counts_pred.shape[1]} genes, "
                f"fewer than target_gene_count={self.config.target_gene_count}"
            )

        # Select the configured highly variable genes.
        counts_256 = counts_pred[:, self.selected_gene_indices]

        if self.config.clip_negative:
            counts_256[counts_256 < 0] = 0.0

        # Restore filtered predictions to the full 16x16 grid when needed.
        full_counts = np.zeros(
            (
                self.config.grid_size * self.config.grid_size,
                self.config.target_gene_count,
            ),
            dtype=np.float32,
        )
        full_counts[keep_mask, :] = counts_256

        st_matrix = full_counts.reshape(
            self.config.grid_size,
            self.config.grid_size,
            self.config.target_gene_count,
        )

        st_matrix = np.asarray(st_matrix, dtype=self.config.dtype)

        self._validate_output(st_matrix, image_path=image_path)
        return st_matrix

    def predict_from_he_matrix(self, he_matrix: np.ndarray) -> np.ndarray:
        """
        Generate pseudo-ST from an H&E array.

        Args:
            he_matrix: RGB array with shape [3, H, W], typically
                [3, 1024, 1024].

        DeepSpot inference is based on image_path and pyvips, so the NumPy
        array is temporarily written to PNG before calling
        predict_from_image_path.
        """
        if not isinstance(he_matrix, np.ndarray):
            raise TypeError(f"he_matrix must be np.ndarray, got {type(he_matrix)}")

        if he_matrix.ndim != 3 or he_matrix.shape[0] != 3:
            raise ValueError(
                f"he_matrix must have shape [3, H, W], got {he_matrix.shape}"
            )

        he = he_matrix
        if he.dtype != np.uint8:
            he = he.astype(np.float32)
            if he.max() <= 1.5:
                he = he * 255.0
            he = np.clip(he, 0, 255).astype(np.uint8)

        he_hwc_rgb = np.transpose(he, (1, 2, 0))
        he_hwc_bgr = cv2.cvtColor(he_hwc_rgb, cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ok = cv2.imwrite(tmp_path, he_hwc_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write temporary H&E image: {tmp_path}")

            return self.predict_from_image_path(tmp_path)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _validate_output(self, st_matrix: np.ndarray, image_path: str = "") -> None:
        expected_shape = (
            self.config.grid_size,
            self.config.grid_size,
            self.config.target_gene_count,
        )
        if st_matrix.shape != expected_shape:
            raise ValueError(
                f"Unexpected ST_matrix shape: {st_matrix.shape}, "
                f"expected={expected_shape}, image={image_path}"
            )

        if not np.isfinite(st_matrix).all():
            n_nan = int(np.isnan(st_matrix).sum())
            n_inf = int(np.isinf(st_matrix).sum())
            raise ValueError(
                f"ST_matrix contains NaN/Inf: nan={n_nan}, "
                f"inf={n_inf}, image={image_path}"
            )

    def get_gene_names(self) -> List[str]:
        """
        Return the names of the selected output genes.
        """
        return list(self.selected_gene_names)

    def describe(self) -> dict:
        """
        Return key DeepSpot wrapper settings for preprocessing reports.
        """
        return {
            "deepspot_repo_path": self.config.deepspot_repo_path,
            "model_weights_path": self.config.model_weights_path,
            "model_hparam_path": self.config.model_hparam_path,
            "gene_info_csv_path": self.config.gene_info_csv_path,
            "morphology_model_path": self.config.morphology_model_path,
            "device": str(self.device),
            "image_feature_model": self.image_feature_model,
            "feature_dim": self.feature_dim,
            "grid_size": self.config.grid_size,
            "target_gene_count": self.config.target_gene_count,
            "n_mini_tiles": self.config.n_mini_tiles,
            "neighbor_radius": self.config.neighbor_radius,
            "filter_white": self.config.filter_white,
            "white_cutoff": self.config.white_cutoff,
            "clip_negative": self.config.clip_negative,
            "selected_gene_names_head": self.selected_gene_names[:10],
        }


def save_st_matrix_debug_json(st_matrix: np.ndarray, save_path: str) -> None:
    """
    Save numerical summary statistics for one ST_matrix.
    """
    report = {
        "shape": list(st_matrix.shape),
        "dtype": str(st_matrix.dtype),
        "min": float(np.min(st_matrix)),
        "max": float(np.max(st_matrix)),
        "mean": float(np.mean(st_matrix)),
        "std": float(np.std(st_matrix)),
        "q01": float(np.quantile(st_matrix, 0.01)),
        "q50": float(np.quantile(st_matrix, 0.50)),
        "q99": float(np.quantile(st_matrix, 0.99)),
        "nan_count": int(np.isnan(st_matrix).sum()),
        "inf_count": int(np.isinf(st_matrix).sum()),
        "nonzero_ratio": float(np.mean(st_matrix != 0)),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)