import os

import numpy as np
import pandas as pd

from ur import config as cfg


TISSUE_CSV = os.path.join(cfg.OUTPUT_DIR, "tissue_qc", "tissue_fraction_per_patch.csv")
FUSION_PRED_CSV = os.path.join(cfg.OUTPUT_DIR, "evaluation", "best_test", "predictions.csv")
UR1_PRED_CSV = "path/to/ur1_ablation/evaluation/best_test/predictions.csv"
UR2_PRED_CSV = "path/to/ur2_ablation/evaluation/best_test/predictions.csv"
OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "tissue_stratified_evaluation")

THRESHOLDS = [0.01, 0.05, 0.10, 0.25]
CONCEPTS = ["DAPI", "CD3", "panCK"]
EPS = 1e-12


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < EPS:
        return np.nan
    return float(np.sum(x * y) / denom)


def r2_score(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    if ss_tot < EPS:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def basic_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {
        "mse": mse, "mae": mae, "rmse": rmse,
        "pearson": pearson(pred, target), "r2": r2_score(pred, target),
        "pred_mean": float(np.mean(pred)), "target_mean": float(np.mean(target)),
        "pred_std": float(np.std(pred)), "target_std": float(np.std(target))
    }


def load_prediction_csv(path, model_name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{model_name} prediction CSV not found:\n{path}")

    df = pd.read_csv(path)
    if "sample_name" not in df.columns:
        raise KeyError(f"{model_name}: sample_name column missing.\nColumns: {list(df.columns)}")

    required = []
    for c in CONCEPTS:
        required.extend([f"target_{c}", f"pred_{c}"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{model_name}: missing columns: {missing}\nAvailable: {list(df.columns)}")

    if df["sample_name"].duplicated().any():
        dup = df.loc[df["sample_name"].duplicated(), "sample_name"].tolist()[:10]
        raise RuntimeError(f"{model_name}: duplicated sample names: {dup}")

    print(f"{model_name:10s} predictions ready | N={len(df)}")
    return df


def load_tissue_csv():
    if not os.path.isfile(TISSUE_CSV):
        raise FileNotFoundError(f"Tissue CSV not found:\n{TISSUE_CSV}")

    df = pd.read_csv(TISSUE_CSV)
    required = ["split", "sample_name", "tissue_fraction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Tissue CSV missing columns: {missing}")

    df = df[df["split"].str.lower() == "test"].copy()
    if df["sample_name"].duplicated().any():
        raise RuntimeError("Duplicated sample names in tissue CSV.")

    print(f"Tissue QC test samples ready | N={len(df)}")
    return df


def align_model_with_tissue(pred_df, tissue_df, model_name):
    tissue_names = set(tissue_df["sample_name"])
    pred_names = set(pred_df["sample_name"])
    missing_tissue = pred_names - tissue_names
    missing_pred = tissue_names - pred_names
    if missing_tissue or missing_pred:
        raise RuntimeError(
            f"{model_name} sample mismatch\n"
            f"prediction-only={len(missing_tissue)}\n"
            f"tissue-only={len(missing_pred)}\n"
            f"prediction-only examples={list(missing_tissue)[:10]}\n"
            f"tissue-only examples={list(missing_pred)[:10]}"
        )

    merged = pred_df.merge(
        tissue_df[["sample_name", "tissue_fraction", "near_white_ratio", "exact_white_ratio"]],
        on="sample_name", how="inner", validate="one_to_one"
    )
    merged = merged.sort_values("sample_name").reset_index(drop=True)
    return merged


def evaluate_subset(df, model_name, subset_name):
    n = len(df)
    if n == 0:
        return []

    rows = []
    pred_all = []
    target_all = []
    for c in CONCEPTS:
        pred_all.append(df[f"pred_{c}"].values)
        target_all.append(df[f"target_{c}"].values)

    pred_all = np.stack(pred_all, axis=1)
    target_all = np.stack(target_all, axis=1)
    diff = pred_all - target_all
    overall_mse = float(np.mean(diff ** 2))
    overall_mae = float(np.mean(np.abs(diff)))
    overall_rmse = float(np.sqrt(overall_mse))

    rows.append({
        "model": model_name, "subset": subset_name, "concept": "Overall", "N": n,
        "mse": overall_mse, "mae": overall_mae, "rmse": overall_rmse,
        "pearson": np.nan, "r2": np.nan,
        "mean_tissue_fraction": float(df["tissue_fraction"].mean()),
        "median_tissue_fraction": float(df["tissue_fraction"].median())
    })

    for c in CONCEPTS:
        pred = df[f"pred_{c}"].values
        target = df[f"target_{c}"].values
        m = basic_metrics(pred, target)
        rows.append({
            "model": model_name, "subset": subset_name, "concept": c, "N": n, **m,
            "mean_tissue_fraction": float(df["tissue_fraction"].mean()),
            "median_tissue_fraction": float(df["tissue_fraction"].median())
        })
    return rows


def build_all_subsets(df):
    """Return the full set, low-tissue subsets, and retained subsets."""
    subsets = {"Full": df}
    for thr in THRESHOLDS:
        pct = int(thr * 100)
        low = df[df["tissue_fraction"] < thr].copy()
        retained = df[df["tissue_fraction"] >= thr].copy()
        subsets[f"Tissue<{pct}%"] = low
        subsets[f"Tissue>={pct}%"] = retained
    return subsets


def print_overall_table(result_df):
    print()
    print("=" * 100)
    print("OVERALL MSE BY TISSUE STRATUM")
    print("=" * 100)

    order = ["Full"]
    for thr in THRESHOLDS:
        pct = int(thr * 100)
        order.extend([f"Tissue<{pct}%", f"Tissue>={pct}%"])

    for subset in order:
        temp = result_df[(result_df["subset"] == subset) & (result_df["concept"] == "Overall")]
        if len(temp) == 0:
            continue
        n = int(temp["N"].iloc[0])
        print()
        print(f"{subset} | N={n}")
        for model in ["Fusion", "UR1-only", "UR2-only"]:
            row = temp[temp["model"] == model]
            if len(row) == 0:
                continue
            mse = row["mse"].iloc[0]
            mae = row["mae"].iloc[0]
            print(f"  {model:10s} | MSE={mse:.6f} | MAE={mae:.6f}")


def print_concept_table(result_df, subset_name):
    print()
    print("=" * 100)
    print(f"PER-CONCEPT | {subset_name}")
    print("=" * 100)

    for model in ["Fusion", "UR1-only", "UR2-only"]:
        print()
        print(model)
        temp = result_df[(result_df["subset"] == subset_name) &
                         (result_df["model"] == model) &
                         (result_df["concept"] != "Overall")]
        if len(temp) == 0:
            print("  No samples")
            continue

        for _, row in temp.iterrows():
            r = row["pearson"]
            r2 = row["r2"]
            r_str = f"{r:.4f}" if np.isfinite(r) else "NA"
            r2_str = f"{r2:.4f}" if np.isfinite(r2) else "NA"
            print(f"  {row['concept']:5s} | MSE={row['mse']:.6f} | MAE={row['mae']:.6f} | "
                  f"r={r_str} | R2={r2_str}")


def make_comparison_table(result_df):
    overall = result_df[result_df["concept"] == "Overall"].copy()
    pivot = overall.pivot(index="subset", columns="model", values="mse").reset_index()
    for model in ["Fusion", "UR1-only", "UR2-only"]:
        if model not in pivot.columns:
            pivot[model] = np.nan

    pivot["Fusion_vs_UR1_delta"] = pivot["Fusion"] - pivot["UR1-only"]
    pivot["Fusion_vs_UR2_delta"] = pivot["Fusion"] - pivot["UR2-only"]
    pivot["Fusion_vs_UR1_improvement_pct"] = (
        (pivot["UR1-only"] - pivot["Fusion"]) / pivot["UR1-only"] * 100.0
    )
    pivot["Fusion_vs_UR2_improvement_pct"] = (
        (pivot["UR2-only"] - pivot["Fusion"]) / pivot["UR2-only"] * 100.0
    )
    return pivot


def print_comparison_table(comp):
    print()
    print("=" * 100)
    print("FUSION RELATIVE TO ABLATIONS")
    print("=" * 100)
    for _, row in comp.iterrows():
        subset = row["subset"]
        f = row["Fusion"]
        u1 = row["UR1-only"]
        u2 = row["UR2-only"]
        d1 = row["Fusion_vs_UR1_improvement_pct"]
        d2 = row["Fusion_vs_UR2_improvement_pct"]
        print(f"{subset:12s} | Fusion={f:.6f} | UR1={u1:.6f} | UR2={u2:.6f} | "
              f"vs UR1={d1:+7.2f}% | vs UR2={d2:+7.2f}%")


def make_per_patch_error_table(model_dfs):
    base = model_dfs["Fusion"][["sample_name", "tissue_fraction", "near_white_ratio"]].copy()
    for model_name, df in model_dfs.items():
        short = model_name.lower().replace("-", "_")
        merged = df[["sample_name"] + [f"target_{c}" for c in CONCEPTS] +
                    [f"pred_{c}" for c in CONCEPTS]].copy()
        for c in CONCEPTS:
            merged[f"{short}_{c}_abs_error"] = np.abs(merged[f"pred_{c}"] - merged[f"target_{c}"])
        keep = ["sample_name"] + [f"{short}_{c}_abs_error" for c in CONCEPTS]
        merged = merged[keep]
        base = base.merge(merged, on="sample_name", how="inner", validate="one_to_one")
    return base


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 100)
    print("Loading tissue QC and predictions")
    print("=" * 100)

    tissue_df = load_tissue_csv()
    prediction_sources = {
        "Fusion": FUSION_PRED_CSV,
        "UR1-only": UR1_PRED_CSV,
        "UR2-only": UR2_PRED_CSV
    }
    model_dfs = {}
    for model_name, path in prediction_sources.items():
        pred_df = load_prediction_csv(path, model_name)
        merged = align_model_with_tissue(pred_df, tissue_df, model_name)
        model_dfs[model_name] = merged

    reference_names = model_dfs["Fusion"]["sample_name"].tolist()
    for model_name, df in model_dfs.items():
        if df["sample_name"].tolist() != reference_names:
            raise RuntimeError(f"Sample order mismatch after alignment: {model_name}")

    for c in CONCEPTS:
        ref = model_dfs["Fusion"][f"target_{c}"].values
        for model_name in ["UR1-only", "UR2-only"]:
            other = model_dfs[model_name][f"target_{c}"].values
            if not np.allclose(ref, other, atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"Target mismatch: Fusion vs {model_name} for {c}")

    print()
    print("All sample names aligned.")
    print("All concept targets identical.")
    print()

    all_rows = []
    for model_name, df in model_dfs.items():
        subsets = build_all_subsets(df)
        for subset_name, subset_df in subsets.items():
            rows = evaluate_subset(subset_df, model_name, subset_name)
            all_rows.extend(rows)
    result_df = pd.DataFrame(all_rows)

    metrics_path = os.path.join(OUTPUT_DIR, "tissue_stratified_metrics.csv")
    result_df.to_csv(metrics_path, index=False)
    comparison = make_comparison_table(result_df)
    comparison_path = os.path.join(OUTPUT_DIR, "fusion_vs_ablation_by_tissue.csv")
    comparison.to_csv(comparison_path, index=False)
    error_df = make_per_patch_error_table(model_dfs)
    error_path = os.path.join(OUTPUT_DIR, "per_patch_errors.csv")
    error_df.to_csv(error_path, index=False)

    print_overall_table(result_df)
    print_comparison_table(comparison)
    print_concept_table(result_df, "Full")
    for thr in THRESHOLDS:
        pct = int(thr * 100)
        print_concept_table(result_df, f"Tissue<{pct}%")
        print_concept_table(result_df, f"Tissue>={pct}%")

    print()
    print("=" * 100)
    print("Finished.")
    print(f"Metrics:    {metrics_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Per-patch:  {error_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()