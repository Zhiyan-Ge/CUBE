import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fig3_common_v3 import as_bool, save_figure
from fig3_paths_v3 import FIG3_RESULT_DIR, FIG3_SOURCE_DIR, FINETUNING_PATCH_SPLIT


def main():
    data = pd.read_csv(FINETUNING_PATCH_SPLIT)
    data["used_for_training"] = as_bool(data["used_for_training"])
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIG3_SOURCE_DIR / "fig3c_spatial_split.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    split_order = ["train", "val", "test", "unused"]
    labels = {"train": "Train", "val": "Validation", "test": "Test", "unused": "Buffer / unused"}

    for split in split_order:
        part = data.loc[data["split"] == split]
        ax.scatter(
            part["block_col"], -part["block_row"], s=42,
            label=f"{labels[split]} (n={len(part)})",
        )

    used = data.loc[data["used_for_training"]]
    ax.scatter(
        used["block_col"], -used["block_row"], s=88,
        facecolors="none", edgecolors="black", linewidths=0.9,
        label=f"Decoder-adaptation subset (n={len(used)})",
    )

    ax.set_xlabel("Spatial block column")
    ax.set_ylabel("Spatial block row")
    ax.set_title("Visium HD contiguous spatial holdout")
    ax.set_aspect("equal")

    # Keep legend and explanatory text outside the data region.
    ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0, fontsize=9,
    )
    fig.text(
        0.10, 0.035,
        "Train, validation, and test regions are separated by unused buffer columns. "
        "All real-ST adaptation freezes the H&E→UR2 encoder and updates only the ST decoder.",
        ha="left", va="bottom", fontsize=9,
    )

    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.13, top=0.90)
    save_figure(fig, FIG3_RESULT_DIR / "fig3c_visium_hd_spatial_split")
    plt.close(fig)


if __name__ == "__main__":
    main()
