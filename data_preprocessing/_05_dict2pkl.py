#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUBE sample packaging utilities.

Each preprocessed sample stores:
    1. he_matrix_512:         float16 [3, 512, 512]
    2. mihc_matrix_512:       float16 [3, 512, 512]
    3. he_matrix_256:         float16 [3, 256, 256]
    4. st_matrix:             float32 [16, 16, 256]
    5. concept_scores:        float32 [3]
    6. sample_name:           str
    7. normalized_mega_coord: float32 [2]
    8. patch_grid_coord:      int64 [2]
    9. metadata:              dict

CUBE no longer uses sub-patch sliding-window augmentation, so
normalized_sub_coord is not stored. The previous st_embedding field is replaced
by the spatial pseudo-ST matrix st_matrix.
"""

import os
import pickle
import numpy as np


IMAGE_512_SHAPE = (3, 512, 512)
IMAGE_256_SHAPE = (3, 256, 256)
ST_SHAPE = (16, 16, 256)


def create_sample_dict(
    normalized_mega_coord,
    patch_grid_coord,
    he_matrix_512: np.ndarray,
    mihc_matrix_512: np.ndarray,
    he_matrix_256: np.ndarray,
    concept_scores,
    st_matrix: np.ndarray,
    sample_name: str,
    metadata: dict = None,
) -> dict:
    """
    Convert one preprocessed sample into the standard CUBE dictionary.
    """
    normalized_mega_coord = np.asarray(normalized_mega_coord, dtype=np.float32)
    patch_grid_coord = np.asarray(patch_grid_coord, dtype=np.int64)
    he_matrix_512 = np.asarray(he_matrix_512, dtype=np.float16)
    mihc_matrix_512 = np.asarray(mihc_matrix_512, dtype=np.float16)
    he_matrix_256 = np.asarray(he_matrix_256, dtype=np.float16)
    concept_scores = np.asarray(concept_scores, dtype=np.float32)
    st_matrix = np.asarray(st_matrix, dtype=np.float32)

    assert normalized_mega_coord.shape == (2,), (
        f"Unexpected normalized_mega_coord shape: "
        f"{normalized_mega_coord.shape}, expected=(2,)"
    )
    assert patch_grid_coord.shape == (2,), (
        f"Unexpected patch_grid_coord shape: "
        f"{patch_grid_coord.shape}, expected=(2,)"
    )
    assert he_matrix_512.shape == IMAGE_512_SHAPE, (
        f"Unexpected he_matrix_512 shape: "
        f"{he_matrix_512.shape}, expected={IMAGE_512_SHAPE}"
    )
    assert mihc_matrix_512.shape == IMAGE_512_SHAPE, (
        f"Unexpected mihc_matrix_512 shape: "
        f"{mihc_matrix_512.shape}, expected={IMAGE_512_SHAPE}"
    )
    assert he_matrix_256.shape == IMAGE_256_SHAPE, (
        f"Unexpected he_matrix_256 shape: "
        f"{he_matrix_256.shape}, expected={IMAGE_256_SHAPE}"
    )
    assert concept_scores.shape == (3,), (
        f"Unexpected concept_scores shape: "
        f"{concept_scores.shape}, expected=(3,)"
    )
    assert st_matrix.shape == ST_SHAPE, (
        f"Unexpected st_matrix shape: {st_matrix.shape}, expected={ST_SHAPE}"
    )

    return {
        "he_matrix_512": he_matrix_512,
        "mihc_matrix_512": mihc_matrix_512,
        "he_matrix_256": he_matrix_256,
        "st_matrix": st_matrix,
        "concept_scores": concept_scores,
        "sample_name": str(sample_name),
        "normalized_mega_coord": normalized_mega_coord,
        "patch_grid_coord": patch_grid_coord,
        "metadata": {} if metadata is None else metadata,
    }


def convert_to_preprocessed_pickle(sample_dict: dict, save_path: str) -> dict:
    """
    Save a standard CUBE sample dictionary as a pickle file.
    """
    final_pkl_dict = {
        "he_matrix_512": np.asarray(
            sample_dict["he_matrix_512"], dtype=np.float16
        ),
        "mihc_matrix_512": np.asarray(
            sample_dict["mihc_matrix_512"], dtype=np.float16
        ),
        "he_matrix_256": np.asarray(
            sample_dict["he_matrix_256"], dtype=np.float16
        ),
        "st_matrix": np.asarray(
            sample_dict["st_matrix"], dtype=np.float32
        ),
        "concept_scores": np.asarray(
            sample_dict["concept_scores"], dtype=np.float32
        ),
        "sample_name": str(sample_dict["sample_name"]),
        "normalized_mega_coord": np.asarray(
            sample_dict["normalized_mega_coord"], dtype=np.float32
        ),
        "patch_grid_coord": np.asarray(
            sample_dict["patch_grid_coord"], dtype=np.int64
        ),
        "metadata": sample_dict["metadata"],
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(final_pkl_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return final_pkl_dict


def save_sample_as_pickle(
    normalized_mega_coord,
    patch_grid_coord,
    he_matrix_512: np.ndarray,
    mihc_matrix_512: np.ndarray,
    he_matrix_256: np.ndarray,
    concept_scores,
    st_matrix: np.ndarray,
    save_path: str,
    sample_name: str,
    metadata: dict = None,
) -> dict:
    """
    Create and save one standard CUBE preprocessing sample.
    """
    sample_dict = create_sample_dict(
        normalized_mega_coord=normalized_mega_coord,
        patch_grid_coord=patch_grid_coord,
        he_matrix_512=he_matrix_512,
        mihc_matrix_512=mihc_matrix_512,
        he_matrix_256=he_matrix_256,
        concept_scores=concept_scores,
        st_matrix=st_matrix,
        sample_name=sample_name,
        metadata=metadata,
    )

    return convert_to_preprocessed_pickle(
        sample_dict=sample_dict,
        save_path=save_path,
    )


def load_preprocessed_pickle(pkl_path: str) -> dict:
    """
    Load a preprocessed CUBE pickle file.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)