#!/usr/bin/env python3
"""Train independent MLP probes from frozen UR1, UR2 and Fusion to the Fig. 5C IE score."""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset


IE_DIR = Path("./result/04_interface_phenotype/data/ie_score_20um")

UR1_DIRS = {
    split: Path("./models/test_models_3/test11/he_mihc/ur1_he") / split
    for split in ("train", "val", "test")
}
UR2_DIRS = {
    split: Path("./models/test_models_3/test13/he_st/ur2_he") / split
    for split in ("train", "val", "test")
}
FUSION_DIRS = {
    split: Path("./result/02_UMAP/data/fused_ur_fixed_coord") / split
    for split in ("train", "val", "test")
}

OUT_DIR = Path("./result/04_interface_phenotype/result/ie_mlp_probe")

GPU_ID = 3
SEED = 2026
EPOCHS = 80
BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
HIDDEN_DIMS = (128, 64)

REPRESENTATIONS = {
    "UR1": {"dirs": UR1_DIRS, "input_dim": 256},
    "UR2": {"dirs": UR2_DIRS, "input_dim": 256},
    "Fusion": {"dirs": FUSION_DIRS, "input_dim": 512},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IEProbeDataset(Dataset):
    def __init__(self, feature_dir, score_csv, representation):
        scores = pd.read_csv(score_csv)
        scores = scores[np.isfinite(scores["ie_score"])].reset_index(drop=True)

        self.feature_dir = feature_dir
        self.representation = representation
        self.sample_names = scores["sample_name"].tolist()
        self.targets = scores["ie_score"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        feature = np.load(self.feature_dir / f"{name}.npy").astype(np.float32)

        if self.representation == "UR1":
            feature = feature.mean(axis=(1, 2))
        elif self.representation == "UR2":
            feature = feature.mean(axis=(1, 2))
        else:
            feature = feature.reshape(-1)

        return {
            "feature": torch.from_numpy(feature),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "sample_name": name,
        }


class IEProbeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_loaders(representation, feature_dirs):
    datasets = {
        split: IEProbeDataset(
            feature_dirs[split],
            IE_DIR / f"ie_score_{split}.csv",
            representation,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }

    return datasets, loaders


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    n = 0

    for batch in loader:
        x = batch["feature"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)

        if training:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

    return loss_sum / n


def predict(model, loader, device):
    model.eval()

    names = []
    targets = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            x = batch["feature"].to(device, non_blocking=True)
            pred = model(x).cpu().numpy()

            names.extend(batch["sample_name"])
            targets.extend(batch["target"].numpy())
            predictions.extend(pred)

    return (
        names,
        np.asarray(targets, dtype=np.float32),
        np.asarray(predictions, dtype=np.float32),
    )


def calculate_metrics(y_true, y_pred):
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def plot_history(history, out_path, representation):
    epochs = np.arange(1, len(history["train"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train"], label="Train")
    plt.plot(epochs, history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{representation} IE probe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_probe(representation, config, device):
    print("=" * 72)
    print(f"Training independent IE probe: {representation}")
    print("=" * 72)

    rep_out = OUT_DIR / representation.lower()
    rep_out.mkdir(parents=True, exist_ok=True)

    datasets, loaders = build_loaders(representation, config["dirs"])

    model = IEProbeMLP(
        input_dim=config["input_dim"],
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = 0
    best_path = rep_out / "best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        val_loss = run_epoch(model, loaders["val"], device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "representation": representation,
                    "input_dim": config["input_dim"],
                },
                best_path,
            )

        print(
            f"{representation} | epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    names, y_true, y_pred = predict(model, loaders["test"], device)
    metrics = calculate_metrics(y_true, y_pred)

    pd.DataFrame({
        "sample_name": names,
        "ie_score": y_true,
        "prediction": y_pred,
    }).to_csv(rep_out / "test_predictions.csv", index=False)

    pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_mse": history["train"],
        "val_mse": history["val"],
    }).to_csv(rep_out / "training_history.csv", index=False)

    plot_history(history, rep_out / "training_curve.pdf", representation)

    report = {
        "representation": representation,
        "train_samples": len(datasets["train"]),
        "val_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "input_dim": config["input_dim"],
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_metrics": metrics,
    }

    with open(rep_out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"{representation} test | "
        f"Pearson={metrics['pearson']:.4f} | "
        f"Spearman={metrics['spearman']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"MAE={metrics['mae']:.4f} | "
        f"MSE={metrics['mse']:.4f}"
    )

    return {
        "representation": representation,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        **metrics,
    }


def plot_comparison(results):
    df = pd.DataFrame(results)

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(7, 5))
    plt.bar(x - width, df["pearson"], width, label="Pearson")
    plt.bar(x, df["spearman"], width, label="Spearman")
    plt.bar(x + width, df["r2"], width, label="R²")
    plt.xticks(x, df["representation"])
    plt.ylabel("Test performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ie_probe_comparison.pdf", dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    results = []

    for i, (representation, config) in enumerate(REPRESENTATIONS.items()):
        set_seed(SEED + i)
        results.append(train_probe(representation, config, device))

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUT_DIR / "ie_probe_comparison.csv", index=False)
    plot_comparison(results)

    print()
    print("=" * 72)
    print("FINAL TEST COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
