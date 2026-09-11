#!/usr/bin/env python3
"""Train six Fig. 5D MLP probes for two real-ST KEGG program scores."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

SCORE_CSV = Path("./result/05_real_st_program/data/program_scores/fig5d_program_scores.csv")
UR1_DIR = Path("./result/05_real_st_program/data/visium_hd_features/ur1")
UR2_DIR = Path("./result/01_real_st/data/HD/visium_hd_cube_finetune/prepared/ur2_features")
FUSION_DIR = Path("./result/05_real_st_program/data/visium_hd_features/fusion_fixed_coord")
OUT_DIR = Path("./result/05_real_st_program/result/program_probes")

TARGETS = {
    "ECM": "ecm_receptor_score",
    "Complement": "complement_coagulation_score",
}
REPRESENTATIONS = ("UR1", "UR2", "Fusion")

GPU_ID = 3
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
SEED = 2026


class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_representation(patch_ids, representation):
    directory = {"UR1": UR1_DIR, "UR2": UR2_DIR, "Fusion": FUSION_DIR}[representation]
    features = []
    for patch_id in patch_ids:
        x = np.load(directory / f"{patch_id}.npy").astype(np.float32)
        if representation != "Fusion":
            x = x.mean(axis=(1, 2))
        features.append(x)
    return np.stack(features)


def make_loader(x, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, x, y, device):
    model.eval()
    with torch.inference_mode():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
    pearson = float(pearsonr(y, pred).statistic)
    spearman = float(spearmanr(y, pred).statistic)
    return pred, pearson, spearman, r2, mae, mse


def train_probe(x, y, train_mask, val_mask, test_mask, input_dim, checkpoint_path, device):
    model = Probe(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(x[train_mask], y[train_mask], True)

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        _, _, _, _, _, val_mse = evaluate(model, x[val_mask], y[val_mask], device)
        train_mse = float(np.mean(losses))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_pred, pearson, spearman, r2, mae, mse = evaluate(model, x[test_mask], y[test_mask], device)
    all_pred, _, _, _, _, _ = evaluate(model, x, y, device)
    return history, all_pred, test_pred, best_epoch, best_val, pearson, spearman, r2, mae, mse


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    metadata = pd.read_csv(SCORE_CSV).reset_index(drop=True)
    patch_ids = metadata["patch_id"].astype(str).tolist()
    train_mask = metadata["split"].eq("train").to_numpy()
    val_mask = metadata["split"].eq("val").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    features = {rep: load_representation(patch_ids, rep) for rep in REPRESENTATIONS}
    metrics, histories = [], []
    predictions = metadata[["patch_id", "block_row", "block_col", "filtered_bins", "filtered_fraction", "split"]].copy()

    for target_name, target_column in TARGETS.items():
        y = metadata[target_column].to_numpy(dtype=np.float32)
        predictions[f"{target_name.lower()}_true"] = y

        for rep in REPRESENTATIONS:
            x = features[rep]
            checkpoint = checkpoint_dir / f"{target_name.lower()}_{rep.lower()}.pt"
            history, all_pred, _, best_epoch, best_val, pearson, spearman, r2, mae, mse = train_probe(
                x, y, train_mask, val_mask, test_mask, x.shape[1], checkpoint, device
            )

            predictions[f"{target_name.lower()}_{rep.lower()}_pred"] = all_pred
            metrics.append({
                "target": target_name, "representation": rep,
                "best_epoch": best_epoch, "best_val_mse": best_val,
                "test_pearson": pearson, "test_spearman": spearman,
                "test_r2": r2, "test_mae": mae, "test_mse": mse,
            })
            for row in history:
                histories.append({"target": target_name, "representation": rep, **row})

            print(
                f"{target_name:10s} | {rep:6s} | best={best_epoch:02d} "
                f"| val={best_val:.4f} | test P={pearson:.4f} "
                f"| S={spearman:.4f} | R2={r2:.4f}"
            )

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT_DIR / "fig5d_probe_metrics.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT_DIR / "fig5d_probe_training_history.csv", index=False)
    predictions.to_csv(OUT_DIR / "fig5d_probe_predictions.csv", index=False)

    print("\n" + "=" * 80)
    print(metrics.to_string(index=False))
    print(f"\nsaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
