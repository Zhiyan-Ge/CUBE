import json
from pathlib import Path

import numpy as np
import pandas as pd

from fig3_paths_v3 import (
    DEEPSPOT_HVG,
    DEEPSPOT_SUMMARY,
    FINETUNING_SUMMARY,
    GENE_METRICS,
    RESET_HEAD_SUMMARY,
)

METHOD_LABELS = {
    "deepspot": "DeepSpot",
    "zero_shot": "CUBE zero-shot",
    "pretrained": "CUBE pretrained-decoder FT",
    "scratch": "CUBE scratch-decoder FT",
    "reset_head": "CUBE reset-head FT",
}


def as_bool(series):
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def save_figure(fig, output_stem, dpi=300):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")


def load_pseudost_gene_names():
    genes = pd.read_csv(DEEPSPOT_HVG)
    predicted = genes.loc[as_bool(genes["isPredicted"])].copy()
    predicted = predicted.sort_values("highly_variable_rank").head(256).reset_index(drop=True)
    return predicted["gene_name"].astype(str).tolist()


def pearson_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    num = np.sum(ac * bc, axis=-1)
    den = np.sqrt(np.sum(ac * ac, axis=-1) * np.sum(bc * bc, axis=-1))
    out = np.full(num.shape, np.nan, dtype=np.float64)
    valid = den > 1e-12
    out[valid] = num[valid] / den[valid]
    return np.clip(out, -1.0, 1.0)


def _summary_row(frame, mode=None):
    row = frame.loc[frame["variant"] == "as_predicted"]
    if mode is not None:
        row = row.loc[row["mode"] == mode]
    return row.iloc[0]


def build_real_st_method_summary():
    ft = pd.read_csv(FINETUNING_SUMMARY)
    reset = pd.read_csv(RESET_HEAD_SUMMARY)
    deep = pd.read_csv(DEEPSPOT_SUMMARY)
    records = []

    for key, frame, mode in [
        ("deepspot", deep, None),
        ("zero_shot", ft, "zero_shot"),
        ("pretrained", ft, "pretrained"),
        ("scratch", ft, "scratch"),
        ("reset_head", reset, "reset_head"),
    ]:
        row = _summary_row(frame, mode)
        records.append({
            "method_key": key,
            "method": METHOD_LABELS[key],
            "gene_pearson": float(row["gene_pearson_mean_detected_ge_50"]),
            "gene_spearman": float(row["gene_spearman_mean_detected_ge_50"]),
            "bin_pearson": float(row["bin_pearson_mean"]),
            "global_pearson": float(row["global_pearson"]),
            "valid_bins": int(row["valid_bins"]),
            "evaluated_genes": int(row["evaluated_genes"]),
            "genes_detected_ge_50": int(row["genes_detected_ge_50"]),
        })
    return pd.DataFrame(records)


def load_gene_metric_tables():
    tables = {}
    for key, path in GENE_METRICS.items():
        frame = pd.read_csv(path).sort_values("model_channel").reset_index(drop=True)
        frame["in_visium_hd"] = as_bool(frame["in_visium_hd"])
        tables[key] = frame
    return tables


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
