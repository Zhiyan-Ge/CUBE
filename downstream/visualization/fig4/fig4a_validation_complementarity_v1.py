import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig4_common_v1 import load_predictions, overall_mse, concept_pearsons, save_figure
from fig4_paths_v1 import (
    CONCEPTS,
    MODELS,
    OUTPUT_DIR,
    FUSION_VAL_PRED_CSV,
    UR1_VAL_PRED_CSV,
    UR2_VAL_PRED_CSV,
)


VAL_PATHS = {
    "UR1-only": UR1_VAL_PRED_CSV,
    "UR2-only": UR2_VAL_PRED_CSV,
    "Fusion": FUSION_VAL_PRED_CSV,
}


def main():
    rows = []
    for model in MODELS:
        df = load_predictions(VAL_PATHS[model])
        pearsons = concept_pearsons(df)
        rows.append({
            "model": model,
            "N": len(df),
            "overall_mse": overall_mse(df),
            **{f"{c}_pearson": pearsons[c] for c in CONCEPTS},
        })

    source = pd.DataFrame(rows)
    source.to_csv(OUTPUT_DIR / "fig4a_validation_complementarity_source.csv", index=False)

    mse = source.set_index("model").loc[MODELS, "overall_mse"]
    heat = source.set_index("model").loc[MODELS, [f"{c}_pearson" for c in CONCEPTS]].to_numpy()

    fusion_mse = mse["Fusion"]
    improve_ur1 = (mse["UR1-only"] - fusion_mse) / mse["UR1-only"] * 100
    improve_ur2 = (mse["UR2-only"] - fusion_mse) / mse["UR2-only"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [0.9, 1.2]})

    ax = axes[0]
    x = np.arange(len(MODELS))
    bars = ax.bar(x, mse.to_numpy(), width=0.62)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Overall concept MSE")
    ax.set_title("Validation overall error")
    ax.set_ylim(0, mse.max() * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, mse):
        ax.text(bar.get_x() + bar.get_width() / 2, value + mse.max() * 0.025, f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.text(2, mse.max() * 0.91, f"−{improve_ur1:.1f}% vs UR1\n−{improve_ur2:.1f}% vs UR2",
            ha="center", va="center", fontsize=8.5)

    ax = axes[1]
    im = ax.imshow(heat, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(CONCEPTS)), CONCEPTS)
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.set_title("Validation concept Pearson")
    for i in range(len(MODELS)):
        for j in range(len(CONCEPTS)):
            value = heat[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.72 else "black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Pearson r")

    fig.suptitle("Complementary UR1 and UR2 representations improve concept prediction", fontsize=12, y=1.01)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig4a_validation_complementarity")
    plt.close(fig)

    print(source.to_string(index=False))
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
