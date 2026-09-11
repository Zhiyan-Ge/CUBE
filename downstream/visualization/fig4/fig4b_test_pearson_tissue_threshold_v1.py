import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_common_v1 import load_test_with_tissue, pearson, save_figure
from fig4_paths_v1 import (
    CONCEPTS, MODELS, THRESHOLDS, THRESHOLD_LABELS, OUTPUT_DIR, TISSUE_CSV,
    FUSION_TEST_PRED_CSV, UR1_TEST_PRED_CSV, UR2_TEST_PRED_CSV,
)


TEST_PATHS = {
    "UR1-only": UR1_TEST_PRED_CSV,
    "UR2-only": UR2_TEST_PRED_CSV,
    "Fusion": FUSION_TEST_PRED_CSV,
}

PLOT_CONCEPTS = CONCEPTS + ["Total"]

COLOR_LIMITS = {
    "DAPI": (0.84, 0.94),
    "CD3": (0.60, 0.84),
    "panCK": (0.60, 1.00),
    "Total": (0.82, 0.98),
}


def overall_pearson(df):
    pred = np.concatenate([df["pred_DAPI"].to_numpy(), df["pred_CD3"].to_numpy(), df["pred_panCK"].to_numpy()])
    target = np.concatenate([df["target_DAPI"].to_numpy(), df["target_CD3"].to_numpy(), df["target_panCK"].to_numpy()])
    return pearson(pred, target)


def main():
    model_data = {model: load_test_with_tissue(path, TISSUE_CSV) for model, path in TEST_PATHS.items()}

    rows = []
    for model in MODELS:
        df = model_data[model]
        for threshold, label in zip(THRESHOLDS, THRESHOLD_LABELS):
            subset = df if threshold == 0 else df[df["tissue_fraction"] >= threshold]
            for concept in CONCEPTS:
                rows.append({
                    "model": model,
                    "minimum_tissue_fraction": threshold,
                    "threshold_label": label,
                    "concept": concept,
                    "N": len(subset),
                    "pearson": pearson(subset[f"pred_{concept}"], subset[f"target_{concept}"]),
                })
            rows.append({
                "model": model,
                "minimum_tissue_fraction": threshold,
                "threshold_label": label,
                "concept": "Total",
                "N": len(subset),
                "pearson": overall_pearson(subset),
            })

    source = pd.DataFrame(rows)
    source.to_csv(OUTPUT_DIR / "fig4b_test_pearson_tissue_threshold_source.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8))
    axes = axes.ravel()

    for ax, concept in zip(axes, PLOT_CONCEPTS):
        matrix = np.zeros((len(MODELS), len(THRESHOLDS)), dtype=float)
        for i, model in enumerate(MODELS):
            values = source[(source["model"] == model) & (source["concept"] == concept)]["pearson"].to_numpy()
            matrix[i] = values

        vmin, vmax = COLOR_LIMITS[concept]
        im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_title(concept, fontsize=12)
        ax.set_xticks(np.arange(len(THRESHOLDS)), THRESHOLD_LABELS, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(MODELS)), MODELS)
        ax.set_xlabel("Minimum retained tissue fraction")

        midpoint = (vmin + vmax) / 2
        for i in range(len(MODELS)):
            for j in range(len(THRESHOLDS)):
                value = matrix[i, j]
                text_color = "white" if value < midpoint else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Pearson r", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Held-out test Pearson across tissue-content thresholds", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, OUTPUT_DIR / "fig4b_test_pearson_tissue_threshold")
    plt.close(fig)

    print(source.to_string(index=False))
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()