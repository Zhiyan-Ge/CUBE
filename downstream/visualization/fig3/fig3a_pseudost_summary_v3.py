import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fig3_common_v3 import save_figure
from fig3_paths_v3 import FIG3_RESULT_DIR, FIG3_SOURCE_DIR, HE_ST_EVAL_JSON


def main():
    with HE_ST_EVAL_JSON.open(encoding="utf-8") as f:
        report = json.load(f)
    he = report["HE->ST"]

    rows = [
        ("Global", he["global_pearson"]),
        ("Gene-wise global", he["gene_wise_global_pearson"]["mean"]),
        ("Cross-sample abundance", he["cross_sample_abundance_pearson"]["mean"]),
        ("Within-patch spatial", he["within_patch_spatial_pearson"]["mean"]),
    ]
    source = pd.DataFrame(rows, columns=["metric", "pearson"])
    source["checkpoint_epoch"] = report["checkpoint_epoch"]
    source["test_samples"] = report["num_samples"]
    source["mse"] = he["mse"]
    source["mse_over_zero_baseline"] = he["mse_over_zero_baseline"]
    source["pred_std_over_target_std"] = he["pred_std_over_target_std"]
    FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIG3_SOURCE_DIR / "fig3a_pseudost_summary.csv", index=False)

    # Keep the panel visually focused on the four correlation metrics.
    # MSE/variance statistics remain in the source table and can be reported in the caption.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    y = list(range(len(source)))
    ax.scatter(source["pearson"], y, s=92, zorder=3)
    for i, value in enumerate(source["pearson"]):
        ax.text(value + 0.014, i, f"{value:.3f}", va="center", fontsize=10)

    ax.set_yticks(y, source["metric"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson correlation")
    ax.set_title("H&E to pseudo-ST reconstruction on Test945")
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.tick_params(axis="both", labelsize=10)

    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.17, top=0.87)
    save_figure(fig, FIG3_RESULT_DIR / "fig3a_pseudost_summary")
    plt.close(fig)


if __name__ == "__main__":
    main()
