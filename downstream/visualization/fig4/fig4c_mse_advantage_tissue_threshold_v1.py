import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_common_v1 import load_test_with_tissue, overall_mse, save_figure
from fig4_paths_v1 import (
    MODELS,
    THRESHOLDS,
    THRESHOLD_LABELS,
    OUTPUT_DIR,
    TISSUE_CSV,
    FUSION_TEST_PRED_CSV,
    UR1_TEST_PRED_CSV,
    UR2_TEST_PRED_CSV,
)


TEST_PATHS = {
    "UR1-only": UR1_TEST_PRED_CSV,
    "UR2-only": UR2_TEST_PRED_CSV,
    "Fusion": FUSION_TEST_PRED_CSV,
}


def main():
    model_data = {
        model: load_test_with_tissue(path, TISSUE_CSV)
        for model, path in TEST_PATHS.items()
    }

    rows = []
    for threshold, label in zip(THRESHOLDS, THRESHOLD_LABELS):
        result = {
            "minimum_tissue_fraction": threshold,
            "threshold_label": label,
        }
        for model in MODELS:
            df = model_data[model]
            subset = df if threshold == 0 else df[df["tissue_fraction"] >= threshold]
            result["N"] = len(subset)
            result[f"{model}_mse"] = overall_mse(subset)
        result["UR1_minus_Fusion_MSE"] = result["UR1-only_mse"] - result["Fusion_mse"]
        result["UR2_minus_Fusion_MSE"] = result["UR2-only_mse"] - result["Fusion_mse"]
        rows.append(result)

    source = pd.DataFrame(rows)
    source.to_csv(OUTPUT_DIR / "fig4c_mse_advantage_tissue_threshold_source.csv", index=False)

    x = np.arange(len(THRESHOLDS))
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    ax.axhline(0, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(x, source["UR1_minus_Fusion_MSE"], marker="o", linewidth=1.9, label="UR1-only − Fusion")
    ax.plot(x, source["UR2_minus_Fusion_MSE"], marker="o", linewidth=1.9, label="UR2-only − Fusion")

    ax.set_xticks(x, THRESHOLD_LABELS)
    ax.set_xlabel("Minimum retained tissue fraction")
    ax.set_ylabel("ΔMSE relative to Fusion")
    ax.set_title("Fusion advantage stabilizes in tissue-rich test subsets")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    ymin, ymax = ax.get_ylim()
    ax.text(len(x) - 0.7, ymax * 0.78, "Fusion better", ha="right", va="center", fontsize=9)
    ax.text(len(x) - 0.7, ymin * 0.78, "Single branch better", ha="right", va="center", fontsize=9)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig4c_mse_advantage_tissue_threshold")
    plt.close(fig)

    print(source.to_string(index=False))
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
