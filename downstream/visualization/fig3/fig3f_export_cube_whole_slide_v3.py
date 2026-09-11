import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fig3_paths_v3 import (
    FIG3_CACHE_DIR,
    FIG3F_GENES,
    GENE_MAPPING,
    GPU_ID,
    GROUPS,
    HE_ST_CHECKPOINT,
    MODEL_ROOT,
    PATCH_METADATA,
    PRETRAINED_DECODER,
    RESET_HEAD_DECODER,
    SCRATCH_DECODER,
    ST_STATS_PATH,
    UR2_FEATURE_DIR,
)

BATCH_SIZE = 32
NUM_WORKERS = 4
SAVE_DTYPE = np.float16

METHODS = {
    "zero_shot": None,
    "pretrained": PRETRAINED_DECODER,
    "scratch": SCRATCH_DECODER,
    "reset_head": RESET_HEAD_DECODER,
}


class FeatureDataset(Dataset):
    def __init__(self, patch_ids):
        self.patch_ids = patch_ids

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]
        feature = np.load(UR2_FEATURE_DIR / f"{patch_id}.npy").astype(np.float32)
        return torch.from_numpy(feature), patch_id


def load_original_decoder(device):
    sys.path.insert(0, str(MODEL_ROOT))
    from he_st.st_decoder import STDecoder

    checkpoint = torch.load(HE_ST_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    prefix = "model3.st_decoder."
    decoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    decoder = STDecoder(GROUPS)
    decoder.load_state_dict(decoder_state, strict=True)
    with np.load(ST_STATS_PATH) as stats:
        mean = stats["mean"].astype(np.float32)
        safe_std = stats["safe_std"].astype(np.float32)
    return decoder.to(device).eval(), mean, safe_std


def load_finetuned_decoder(path, device):
    sys.path.insert(0, str(MODEL_ROOT))
    from he_st.st_decoder import STDecoder

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    decoder = STDecoder(GROUPS)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    safe_std = np.asarray(checkpoint["target_safe_std"], dtype=np.float32)
    return decoder.to(device).eval(), mean, safe_std


def main():
    metadata = pd.read_csv(PATCH_METADATA).sort_values(["block_col", "block_row"]).reset_index(drop=True)
    patch_ids = metadata["patch_id"].astype(str).tolist()
    mapping = pd.read_csv(GENE_MAPPING).set_index("gene_name")
    channels = np.asarray([int(mapping.loc[g, "model_channel"]) for g in FIG3F_GENES], dtype=np.int64)
    loader = DataLoader(
        FeatureDataset(patch_ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    FIG3_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for method, checkpoint in METHODS.items():
        if method == "zero_shot":
            decoder, mean, safe_std = load_original_decoder(device)
        else:
            decoder, mean, safe_std = load_finetuned_decoder(checkpoint, device)

        selected = []
        with torch.inference_mode():
            for features, _ in tqdm(loader, desc=f"Fig3F CUBE {method}", dynamic_ncols=True):
                features = features.to(device, non_blocking=True)
                prediction_z = decoder(features)
                prediction = prediction_z.float().cpu().numpy()
                prediction = prediction * safe_std.reshape(1, 1, 1, 256)
                prediction += mean.reshape(1, 1, 1, 256)
                selected.append(prediction[..., channels])

        values = np.concatenate(selected, axis=0)
        np.savez_compressed(
            FIG3_CACHE_DIR / f"fig3f_whole_slide_{method}.npz",
            patch_ids=np.asarray(patch_ids), gene_names=np.asarray(FIG3F_GENES),
            model_channels=channels, predictions_log1p_cp10k=values.astype(SAVE_DTYPE),
        )
        del decoder
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
