import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from fig3_common_v3 import save_figure
from fig3_paths_v3 import FIG3_CACHE_DIR, FIG3_RESULT_DIR


def main():
    with np.load(FIG3_CACHE_DIR / "fig3b_selected_examples.npz") as data:
        he = data["he"]
        target = data["target"]
        prediction = data["prediction"]
        residual = data["residual"]
        gene_name = data["gene_name"].astype(str)
        role = data["selection_role"].astype(str)
        spatial_r = data["spatial_pearson"]

    n_rows = len(he)
    fig = plt.figure(figsize=(11.8, 2.8 * n_rows + 0.6))
    gs = GridSpec(
        n_rows, 5, figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.045, 1.0],
        wspace=0.16, hspace=0.20,
    )

    axes = []
    for row in range(n_rows):
        ax_he = fig.add_subplot(gs[row, 0])
        ax_target = fig.add_subplot(gs[row, 1])
        ax_pred = fig.add_subplot(gs[row, 2])
        cax = fig.add_subplot(gs[row, 3])
        ax_res = fig.add_subplot(gs[row, 4])
        axes.append((ax_he, ax_target, ax_pred, cax, ax_res))

        ax_he.imshow(np.clip(he[row], 0, 1))
        vmin = min(float(target[row].min()), float(prediction[row].min()))
        vmax = max(float(target[row].max()), float(prediction[row].max()))
        im = ax_target.imshow(target[row], vmin=vmin, vmax=vmax, cmap="viridis")
        ax_pred.imshow(prediction[row], vmin=vmin, vmax=vmax, cmap="viridis")

        lim = max(abs(float(residual[row].min())), abs(float(residual[row].max())), 1e-6)
        ax_res.imshow(residual[row], vmin=-lim, vmax=lim, cmap="coolwarm")
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8, length=2)

        if row == 0:
            ax_he.set_title("H&E", fontsize=11)
            ax_target.set_title("Target pseudo-ST", fontsize=11)
            ax_pred.set_title("CUBE prediction", fontsize=11)
            ax_res.set_title("Residual", fontsize=11)

        # Publication-facing label intentionally omits the long patch identifier.
        # Exact sample_index/sample_name remain in fig3b_selected_examples.csv.
        ax_he.text(
            -0.06, 0.5,
            f"{gene_name[row]}\n{role[row]}\nspatial r = {spatial_r[row]:.3f}",
            transform=ax_he.transAxes, ha="right", va="center", fontsize=9,
        )

        for ax in (ax_he, ax_target, ax_pred, ax_res):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Representative H&E to pseudo-ST spatial reconstructions", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.04, top=0.93)
    save_figure(fig, FIG3_RESULT_DIR / "fig3b_pseudost_examples")
    plt.close(fig)


if __name__ == "__main__":
    main()
