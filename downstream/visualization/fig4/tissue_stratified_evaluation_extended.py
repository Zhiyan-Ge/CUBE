import os
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

TISSUE_CSV = "./models/test_models_3/test15/ur_branch/tissue_qc/tissue_fraction_per_patch.csv"

FUSION_PRED_CSV = "./models/test_models_3/test15/ur_branch/evaluation/best_test/predictions.csv"
UR1_PRED_CSV = "./models/ur1_ablation/evaluation/best_test/predictions.csv"
UR2_PRED_CSV = "./models/ur2_ablation/evaluation/best_test/predictions.csv"

OUTPUT_DIR = "./result/visualization/fig4/result/tissue_stratified"


# =============================================================================
# Settings
# =============================================================================

THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.30, 0.50, 0.75]
CONCEPTS = ["DAPI", "CD3", "panCK"]
EPS = 1e-12


# =============================================================================
# Metrics
# =============================================================================

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

    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "pearson": pearson(pred, target),
        "r2": r2_score(pred, target),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "pred_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
    }


# =============================================================================
# Data
# =============================================================================

def load_model_predictions(path, tissue_df):
    df = pd.read_csv(path)

    df = df.merge(
        tissue_df[["sample_name", "tissue_fraction"]],
        on="sample_name",
        how="inner",
    )

    return df.sort_values("sample_name").reset_index(drop=True)


def load_data():
    tissue_df = pd.read_csv(TISSUE_CSV)
    tissue_df = tissue_df[tissue_df["split"].str.lower() == "test"].copy()

    model_dfs = {
        "Fusion": load_model_predictions(FUSION_PRED_CSV, tissue_df),
        "UR1-only": load_model_predictions(UR1_PRED_CSV, tissue_df),
        "UR2-only": load_model_predictions(UR2_PRED_CSV, tissue_df),
    }

    return model_dfs


# =============================================================================
# Stratified evaluation
# =============================================================================

def evaluate_subset(df, model_name, subset_name):
    if len(df) == 0:
        return []

    rows = []

    pred_all = np.stack(
        [df[f"pred_{c}"].values for c in CONCEPTS],
        axis=1,
    )

    target_all = np.stack(
        [df[f"target_{c}"].values for c in CONCEPTS],
        axis=1,
    )

    diff = pred_all - target_all
    overall_mse = float(np.mean(diff ** 2))

    rows.append({
        "model": model_name,
        "subset": subset_name,
        "concept": "Overall",
        "N": len(df),
        "mse": overall_mse,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(overall_mse)),
        "pearson": np.nan,
        "r2": np.nan,
        "mean_tissue_fraction": float(df["tissue_fraction"].mean()),
        "median_tissue_fraction": float(df["tissue_fraction"].median()),
        "pred_mean": np.nan,
        "target_mean": np.nan,
        "pred_std": np.nan,
        "target_std": np.nan,
    })

    for concept in CONCEPTS:
        metrics = basic_metrics(
            df[f"pred_{concept}"].values,
            df[f"target_{concept}"].values,
        )

        rows.append({
            "model": model_name,
            "subset": subset_name,
            "concept": concept,
            "N": len(df),
            **metrics,
            "mean_tissue_fraction": float(df["tissue_fraction"].mean()),
            "median_tissue_fraction": float(df["tissue_fraction"].median()),
        })

    return rows


def build_subsets(df):
    subsets = {"Full": df}

    for threshold in THRESHOLDS:
        pct = int(round(threshold * 100))

        subsets[f"Tissue<{pct}%"] = df[
            df["tissue_fraction"] < threshold
        ].copy()

        subsets[f"Tissue>={pct}%"] = df[
            df["tissue_fraction"] >= threshold
        ].copy()

    return subsets


# =============================================================================
# Comparison table
# =============================================================================

def make_comparison_table(metrics_df):
    overall = metrics_df[
        metrics_df["concept"] == "Overall"
    ].copy()

    pivot = overall.pivot(
        index=["subset", "N"],
        columns="model",
        values="mse",
    ).reset_index()

    pivot["Fusion_vs_UR1_delta"] = (
        pivot["Fusion"] - pivot["UR1-only"]
    )

    pivot["Fusion_vs_UR2_delta"] = (
        pivot["Fusion"] - pivot["UR2-only"]
    )

    pivot["Fusion_vs_UR1_improvement_pct"] = (
        (pivot["UR1-only"] - pivot["Fusion"])
        / pivot["UR1-only"]
        * 100.0
    )

    pivot["Fusion_vs_UR2_improvement_pct"] = (
        (pivot["UR2-only"] - pivot["Fusion"])
        / pivot["UR2-only"]
        * 100.0
    )

    return pivot


# =============================================================================
# Retained-threshold table for Fig4
# =============================================================================

def make_retained_threshold_table(metrics_df):
    rows = []

    for threshold in [0.0] + THRESHOLDS:
        if threshold == 0:
            subset = "Full"
            threshold_pct = 0
        else:
            threshold_pct = int(round(threshold * 100))
            subset = f"Tissue>={threshold_pct}%"

        temp = metrics_df[
            (metrics_df["subset"] == subset)
            & (metrics_df["concept"] == "Overall")
        ]

        row = {
            "minimum_tissue_fraction": threshold,
            "minimum_tissue_percent": threshold_pct,
            "subset": subset,
            "N": int(temp["N"].iloc[0]),
        }

        for model in ["Fusion", "UR1-only", "UR2-only"]:
            model_row = temp[temp["model"] == model].iloc[0]

            row[f"{model}_mse"] = model_row["mse"]
            row[f"{model}_mae"] = model_row["mae"]

        row["Fusion_vs_UR1_improvement_pct"] = (
            (row["UR1-only_mse"] - row["Fusion_mse"])
            / row["UR1-only_mse"]
            * 100.0
        )

        row["Fusion_vs_UR2_improvement_pct"] = (
            (row["UR2-only_mse"] - row["Fusion_mse"])
            / row["UR2-only_mse"]
            * 100.0
        )

        row["UR1_minus_Fusion_MSE"] = (
            row["UR1-only_mse"] - row["Fusion_mse"]
        )

        row["UR2_minus_Fusion_MSE"] = (
            row["UR2-only_mse"] - row["Fusion_mse"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Print
# =============================================================================

def print_summary(metrics_df):
    print()
    print("=" * 100)
    print("TISSUE-STRATIFIED TEST PERFORMANCE")
    print("=" * 100)

    subsets = ["Full"]

    for threshold in THRESHOLDS:
        pct = int(round(threshold * 100))
        subsets.append(f"Tissue>={pct}%")

    for subset in subsets:
        temp = metrics_df[
            (metrics_df["subset"] == subset)
            & (metrics_df["concept"] == "Overall")
        ]

        n = int(temp["N"].iloc[0])

        print()
        print(f"{subset} | N={n}")

        for model in ["Fusion", "UR1-only", "UR2-only"]:
            row = temp[temp["model"] == model].iloc[0]

            print(
                f"  {model:10s} | "
                f"MSE={row['mse']:.6f} | "
                f"MAE={row['mae']:.6f}"
            )


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_dfs = load_data()

    rows = []

    for model_name, df in model_dfs.items():
        subsets = build_subsets(df)

        for subset_name, subset_df in subsets.items():
            rows.extend(
                evaluate_subset(
                    subset_df,
                    model_name,
                    subset_name,
                )
            )

    metrics_df = pd.DataFrame(rows)

    comparison_df = make_comparison_table(metrics_df)
    retained_df = make_retained_threshold_table(metrics_df)

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "tissue_stratified_metrics_extended.csv",
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "fusion_vs_ablation_by_tissue_extended.csv",
    )

    retained_path = os.path.join(
        OUTPUT_DIR,
        "retained_tissue_threshold_summary.csv",
    )

    metrics_df.to_csv(metrics_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    retained_df.to_csv(retained_path, index=False)

    print_summary(metrics_df)

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)
    print(metrics_path)
    print(comparison_path)
    print(retained_path)


if __name__ == "__main__":
    main()