import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fig3_paths_v3 import (
    DEEPSPOT_HPARAMS,
    DEEPSPOT_HVG,
    DEEPSPOT_MODEL_WEIGHTS,
    DEEPSPOT_MORPHOLOGY_WEIGHTS,
    DEEPSPOT_REPO_DIR,
    DEEPSPOT_WRAPPER,
    FIG3_CACHE_DIR,
    FIG3F_GENES,
    GENE_MAPPING,
    NORMALIZED_HD_PATCH_DIR,
    PATCH_METADATA,
)

DEVICE = "cuda"
SAVE_DTYPE = np.float16


def load_wrapper():
    spec = importlib.util.spec_from_file_location("cube_deepspot_wrapper", DEEPSPOT_WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_generator(module):
    config = module.DeepSpotConfig(
        deepspot_repo_path=str(DEEPSPOT_REPO_DIR),
        model_weights_path=str(DEEPSPOT_MODEL_WEIGHTS),
        model_hparam_path=str(DEEPSPOT_HPARAMS),
        gene_info_csv_path=str(DEEPSPOT_HVG),
        morphology_model_path=str(DEEPSPOT_MORPHOLOGY_WEIGHTS),
        device=DEVICE,
        grid_size=16,
        target_gene_count=256,
        spot_diameter_px=64,
        spot_distance_px=64,
        filter_white=False,
        clip_negative=False,
        dtype="float32",
    )
    return module.DeepSpotPseudoSTGenerator(config)


def main():
    metadata = pd.read_csv(PATCH_METADATA).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    patch_ids = metadata["patch_id"].astype(str).tolist()
    mapping = pd.read_csv(GENE_MAPPING).sort_values("model_channel").reset_index(drop=True)
    channels = np.asarray([
        int(mapping.loc[mapping["gene_name"] == gene, "model_channel"].iloc[0])
        for gene in FIG3F_GENES
    ], dtype=np.int64)

    module = load_wrapper()
    generator = make_generator(module)
    expected = mapping["gene_name"].astype(str).tolist()
    if generator.get_gene_names() != expected:
        raise ValueError("DeepSpot output gene order differs from gene_mapping.csv")

    selected = []
    with torch.inference_mode():
        for patch_id in tqdm(patch_ids, desc="Fig3F DeepSpot whole slide", dynamic_ncols=True):
            prediction = generator.predict_from_image_path(str(NORMALIZED_HD_PATCH_DIR / f"{patch_id}.png"))
            selected.append(np.asarray(prediction[..., channels], dtype=np.float32))

    FIG3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIG3_CACHE_DIR / "fig3f_whole_slide_deepspot.npz",
        patch_ids=np.asarray(patch_ids), gene_names=np.asarray(FIG3F_GENES),
        model_channels=channels,
        predictions_log1p_cp10k=np.asarray(selected, dtype=SAVE_DTYPE),
    )


if __name__ == "__main__":
    main()
