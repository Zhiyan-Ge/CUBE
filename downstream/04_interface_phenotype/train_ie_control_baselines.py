#!/usr/bin/env python3
"""Train Fig. 5C control MLPs: simple UR1+UR2 concatenation and abundance concepts."""

import json
import random
from pathlib import Path

import cv2
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
RAW_ROOT = Path("./data/paired_data/he_mihc/raw_data/HEMIT_dataset")

UR1_DIRS = {
    split: Path("./models/test_models_3/test11/he_mihc/ur1_he") / split
    for split in ("train", "val", "test")
}
UR2_DIRS = {
    split: Path("./models/test_models_3/test13/he_st/ur2_he") / split
    for split in ("train", "val", "test")
}

MAIN_RESULT = Path("./result/04_interface_phenotype/result/ie_mlp_probe/ie_probe_comparison.csv")
OUT_DIR = Path("./result/04_interface_phenotype/result/ie_control_baselines")

GPU_ID = 3
SEED = 2026
EPOCHS = 80
BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
HIDDEN_DIMS = (128, 64)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def otsu_fraction(channel):
    threshold, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((channel > threshold).mean())


def load_concepts(path):
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return np.asarray([
        otsu_fraction(image[:, :, 0]),
        otsu_fraction(image[:, :, 1]),
        otsu_fraction(image[:, :, 2]),
    ], dtype=np.float32)


class ConcatDataset(Dataset):
    def __init__(self, split):
        scores = pd.read_csv(IE_DIR / f"ie_score_{split}.csv")
        scores = scores[np.isfinite(scores["ie_score"])].reset_index(drop=True)

        self.ur1_dir = UR1_DIRS[split]
        self.ur2_dir = UR2_DIRS[split]
        self.sample_names = scores["sample_name"].tolist()
        self.targets = scores["ie_score"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]

        ur1 = np.load(self.ur1_dir / f"{name}.npy").astype(np.float32).mean(axis=(1, 2))
        ur2 = np.load(self.ur2_dir / f"{name}.npy").astype(np.float32).mean(axis=(1, 2))
        feature = np.concatenate([ur1, ur2])

        return {
            "feature": torch.from_numpy(feature),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "sample_name": name,
        }


class ConceptDataset(Dataset):
    def __init__(self, split):
        scores = pd.read_csv(IE_DIR / f"ie_score_{split}.csv")
        scores = scores[np.isfinite(scores["ie_score"])].reset_index(drop=True)

        self.sample_names = scores["sample_name"].tolist()
        self.targets = scores["ie_score"].to_numpy(dtype=np.float32)

        label_dir = RAW_ROOT / split / "label"
        self.features = []

        for i, name in enumerate(self.sample_names, 1):
            self.features.append(load_concepts(label_dir / f"{name}.tif"))

            if i % 200 == 0:
                print(f"Concepts {split}: {i}/{len(self.sample_names)}")

        self.features = np.stack(self.features)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        return {
            "feature": torch.from_numpy(self.features[index]),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "sample_name": self.sample_names[index],
        }


class IEProbeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_loaders(dataset_class):
    datasets = {split: dataset_class(split) for split in ("train", "val", "test")}

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

    total_loss = 0.0
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

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n


def predict(model, loader, device):
    model.eval()

    names, targets, predictions = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["feature"].to(device, non_blocking=True)
            pred = model(x).cpu().numpy()

            names.extend(batch["sample_name"])
            targets.extend(batch["target"].numpy())
            predictions.extend(pred)

    return names, np.asarray(targets), np.asarray(predictions)


def calculate_metrics(y_true, y_pred):
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def train_baseline(name, dataset_class, input_dim, device):
    print("=" * 72)
    print(f"Training independent IE baseline: {name}")
    print("=" * 72)

    run_dir = OUT_DIR / name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets, loaders = build_loaders(dataset_class)

    model = IEProbeMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = 0
    best_path = run_dir / "best.pt"

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
                    "baseline": name,
                    "input_dim": input_dim,
                },
                best_path,
            )

        print(
            f"{name} | epoch {epoch:02d}/{EPOCHS} | "
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
    }).to_csv(run_dir / "test_predictions.csv", index=False)

    pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_mse": history["train"],
        "val_mse": history["val"],
    }).to_csv(run_dir / "training_history.csv", index=False)

    report = {
        "representation": name,
        "train_samples": len(datasets["train"]),
        "val_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "input_dim": input_dim,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_metrics": metrics,
    }

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"{name} test | "
        f"Pearson={metrics['pearson']:.4f} | "
        f"Spearman={metrics['spearman']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"MAE={metrics['mae']:.4f} | "
        f"MSE={metrics['mse']:.4f}"
    )

    return {
        "representation": name,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        **metrics,
    }


def plot_combined(df):
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, df["pearson"], width, label="Pearson")
    plt.bar(x, df["spearman"], width, label="Spearman")
    plt.bar(x + width, df["r2"], width, label="R²")
    plt.xticks(x, df["representation"])
    plt.ylabel("Test performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ie_probe_combined_comparison.pdf", dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    results = []

    set_seed(SEED)
    results.append(train_baseline("Concat", ConcatDataset, 512, device))

    set_seed(SEED + 1)
    results.append(train_baseline("Concepts", ConceptDataset, 3, device))

    baseline_df = pd.DataFrame(results)
    baseline_df.to_csv(OUT_DIR / "ie_control_baselines.csv", index=False)

    main_df = pd.read_csv(MAIN_RESULT)
    combined = pd.concat([main_df, baseline_df], ignore_index=True)
    combined.to_csv(OUT_DIR / "ie_probe_combined_comparison.csv", index=False)
    plot_combined(combined)

    print()
    print("=" * 72)
    print("COMBINED TEST COMPARISON")
    print("=" * 72)
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
