import numpy as np
import pandas as pd

from fig4_paths_v1 import CONCEPTS


def load_predictions(path):
    return pd.read_csv(path)


def load_test_with_tissue(pred_path, tissue_path):
    pred = pd.read_csv(pred_path)
    tissue = pd.read_csv(tissue_path)
    if "split" in tissue.columns:
        tissue = tissue[tissue["split"].str.lower() == "test"]
    tissue = tissue[["sample_name", "tissue_fraction"]]
    return pred.merge(tissue, on="sample_name", how="inner")


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def overall_mse(df):
    pred = np.stack([df[f"pred_{c}"].to_numpy() for c in CONCEPTS], axis=1)
    target = np.stack([df[f"target_{c}"].to_numpy() for c in CONCEPTS], axis=1)
    return float(np.mean((pred - target) ** 2))


def concept_pearsons(df):
    return {
        c: pearson(df[f"pred_{c}"].to_numpy(), df[f"target_{c}"].to_numpy())
        for c in CONCEPTS
    }


def save_figure(fig, output_stem):
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
