import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fig3_common_v3 import load_pseudost_gene_names, pearson_rows
from fig3_paths_v3 import (
    FIG3_CACHE_DIR,
    FIG3_SOURCE_DIR,
    GPU_ID,
    GROUPS,
    HE_ST_CHECKPOINT,
    MODEL_ROOT,
    SEED,
    ST_STATS_PATH,
    TEST_PKL_DIR,
)

sys.path.insert(0, str(MODEL_ROOT))
from he_st.dataset import HESTDataset
from he_st.model import HESTBranch

BATCH_SIZE = 16
NUM_WORKERS = 8
SELECTION_QUANTILES = [0.80, 0.55, 0.30]
SELECTION_LABELS = ["strong spatial", "intermediate spatial", "challenging spatial"]
TARGET_STD_QUANTILE = 0.60


def load_model(device):
    checkpoint = torch.load(HE_ST_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model = HESTBranch(GROUPS).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def choose_pairs(spatial_r, target_std):
    threshold = np.nanquantile(target_std, TARGET_STD_QUANTILE)
    valid = np.isfinite(spatial_r) & np.isfinite(target_std) & (target_std >= threshold)
    values = spatial_r[valid]
    targets = [np.nanquantile(values, q) for q in SELECTION_QUANTILES]
    used_samples, used_genes, selected = set(), set(), []

    for label, target in zip(SELECTION_LABELS, targets):
        distance = np.where(valid, np.abs(spatial_r - target), np.inf)
        order = np.argsort(distance, axis=None)
        for flat in order:
            sample_index, gene_index = np.unravel_index(flat, distance.shape)
            if sample_index in used_samples or gene_index in used_genes:
                continue
            selected.append((sample_index, gene_index, label))
            used_samples.add(sample_index)
            used_genes.add(gene_index)
            break
    return selected


def main():
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    dataset = HESTDataset(str(TEST_PKL_DIR), str(ST_STATS_PATH))
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    model = load_model(device)

    all_r, all_std = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Scan pseudo-ST examples", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            pred = model.model3.st_decoder(model.model3.encoder(he)).float().cpu().numpy()
            target = batch["st"].numpy()
            p = pred.transpose(0, 3, 1, 2).reshape(len(pred), 256, -1)
            t = target.transpose(0, 3, 1, 2).reshape(len(target), 256, -1)
            all_r.append(pearson_rows(p, t))
            all_std.append(t.std(axis=-1))

    spatial_r = np.concatenate(all_r, axis=0)
    target_std = np.concatenate(all_std, axis=0)
    selected = choose_pairs(spatial_r, target_std)
    gene_names = load_pseudost_gene_names()

    records = []
    he_images, target_maps, pred_maps, residual_maps = [], [], [], []
    with torch.inference_mode():
        for sample_index, gene_index, label in selected:
            sample = dataset[sample_index]
            he = sample["he"].unsqueeze(0).to(device)
            pred = model.model3.st_decoder(model.model3.encoder(he))[0].float().cpu().numpy()
            target = sample["st"].numpy()
            t = target[:, :, gene_index]
            p = pred[:, :, gene_index]

            he_images.append(sample["he"].numpy().transpose(1, 2, 0))
            target_maps.append(t)
            pred_maps.append(p)
            residual_maps.append(p - t)
            records.append({
                "selection_role": label,
                "sample_index": sample_index,
                "sample_name": str(sample["sample_name"]),
                "gene_index": gene_index,
                "gene_name": gene_names[gene_index],
                "spatial_pearson": float(spatial_r[sample_index, gene_index]),
                "target_std": float(target_std[sample_index, gene_index]),
            })

    FIG3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(FIG3_SOURCE_DIR / "fig3b_selected_examples.csv", index=False)
    np.savez_compressed(
        FIG3_CACHE_DIR / "fig3b_selected_examples.npz",
        he=np.asarray(he_images, dtype=np.float32),
        target=np.asarray(target_maps, dtype=np.float32),
        prediction=np.asarray(pred_maps, dtype=np.float32),
        residual=np.asarray(residual_maps, dtype=np.float32),
        sample_name=np.asarray([r["sample_name"] for r in records]),
        gene_name=np.asarray([r["gene_name"] for r in records]),
        selection_role=np.asarray([r["selection_role"] for r in records]),
        spatial_pearson=np.asarray([r["spatial_pearson"] for r in records], dtype=np.float32),
    )
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
