# CUBE downstream analyses

This directory contains the downstream analyses used to evaluate transferability, latent-space organization, spatial phenotypes, and transcriptome-associated information in CUBE representations.

The directory structure is preserved from the final analysis workflow. Most scripts use repository-root-relative paths such as `./data/`, `./models/`, and `./result/`; unless noted otherwise, run them from the repository root.

## Directory guide

| Directory | Purpose | Main manuscript output |
|---|---|---|
| `01_real_st/` | Visium HD preprocessing, spatial split, frozen-UR2 transfer, decoder adaptation, and DeepSpot comparison | Figure 3c–f |
| `02_UMAP/` | fixed-coordinate fused representation export and UMAP analysis | Figure 5a |
| `03_spatial_cluster/` | K-means clustering and spatial remapping of fused representation states | Figure 5b–c |
| `04_interface_phenotype/` | immune–epithelial enrichment score and representation probing | Figure 5d |
| `05_real_st_program/` | ECM-associated real-ST program definition and representation probing | Figure 5e |
| `visualization/` | manuscript plotting scripts | Figures 2–5 |

These analyses consume trained CUBE checkpoints and/or precomputed UR1/UR2 features. They do not modify the pretrained branch encoders unless explicitly stated in the real-ST decoder-adaptation workflow.

## 01_real_st — transfer to experimentally measured Visium HD

This workflow evaluates whether the H&E-derived UR2 learned from pseudo-ST supervision can transfer to experimentally measured spatial transcriptomics.

The reported experiment uses a colorectal cancer Visium HD section at the 16 μm aggregated-bin resolution. CUBE and DeepSpot are evaluated on the same contiguous spatial holdout. During real-ST adaptation, the pretrained H&E→UR2 encoder remains frozen and only the UR2→ST decoder is optimized.

The main workflow is:

```text
Visium HD raw outputs
    ↓
build_visium_hd_paired_patches.py
    ↓
normalize_visium_hd_all_patches.py
    ↓
prepare_visium_hd_spatial_split_and_ur2.py
    ↓
finetune_visium_hd_cube_decoder.py
    ├── pretrained-decoder initialization
    └── scratch-decoder initialization
    ↓
finetune_visium_hd_cube_reset_head.py
    ↓
evaluate_visium_hd_deepspot_spatial_test.py
    ↓
predict_and_visualize_visium_hd_whole_slide.py
```

The frozen spatial split contains 294 training-region patches, 119 validation patches, 111 held-out test patches, and 47 unused buffer patches. The decoder-adaptation experiments use a fixed subset of 100 training-region patches satisfying the minimum valid-bin criterion.

The three decoder-adaptation settings are:

- **pretrained** — initialize the complete decoder from the pseudo-ST-trained CUBE checkpoint;
- **scratch** — randomly initialize the complete decoder while keeping the H&E→UR2 encoder frozen;
- **reset-head** — retain the pretrained decoder trunk but reinitialize the final output layer before optimizing the full decoder.

All final quantitative real-ST metrics are calculated on the fixed 111-patch held-out test region.

## 02_UMAP — fixed-coordinate fused representation

The fusion model uses global spatial coordinates as an input. To visualize representation geometry without directly plotting absolute-coordinate information, the Figure 5 latent-space analysis re-encodes every HEMIT patch using the same dataset-level mean coordinate.

Run:

```bash
python downstream/02_UMAP/export_fixed_coord_ur.py
python downstream/02_UMAP/run_fixed_coord_umap.py
```

`export_fixed_coord_ur.py` generates fixed-coordinate fused representations for train, validation, and test patches. `run_fixed_coord_umap.py` reduces the 512-dimensional fused representations to 50 principal components and performs UMAP using the frozen analysis settings.

The principal visualization settings are:

```text
PCA components      50
UMAP neighbors      30
UMAP min_dist       0.25
metric              Euclidean
random seed         2026
```

The output is used by Figure 5a and as an input to the spatial clustering workflow.

## 03_spatial_cluster — spatial organization of fused-UR states

This directory examines whether unsupervised states in the fixed-coordinate fused representation form coherent tissue-level spatial patterns when mapped back to their original HEMIT regions.

The retained scripts provide:

- `inspect_hemit_spatial_geometry.py` — reconstructs and checks HEMIT patch-grid geometry;
- `reconstruct_hemit_mihc_regions.py` — reconstructs corresponding mIHC region mosaics;
- `run_fig5b_cluster_selection.py` — evaluates K-means solutions across `K=3...8` on PCA-reduced fixed-coordinate fused representations;
- `plot_fig5b_region_panels.py` — renders spatial region panels from archived cluster assignments.

The manuscript uses an eight-cluster exploratory partition for Figure 5b–c. These clusters are representation states and are **not** treated as annotated pathological classes or pixel-level tissue segmentations.

### Archived intermediate assignment file

The final Figure 5b/c plotting scripts expect:

```text
./result/03_spatial_cluster/result/fig5b_high_resolution_comparison/
    high_resolution_cluster_assignments_with_phenotypes.csv
```

This is an archived intermediate analysis artifact used by the final plotting code. The retained `run_fig5b_cluster_selection.py` produces the K=3–8 cluster-selection outputs but does not recreate this exact final CSV filename. For exact figure reproduction, restore this CSV from the accompanying source-data/research-artifact archive or update the plotting scripts to the corresponding regenerated assignment table after verifying identical sample and cluster labels.

This distinction is kept explicit rather than silently treating exploratory cluster selection and final figure assembly as the same step.

## 04_interface_phenotype — immune–epithelial spatial enrichment

This analysis tests whether frozen CUBE representations retain spatial protein information beyond the three marker-abundance concept scores used during fusion training.

Run in order:

```bash
python downstream/04_interface_phenotype/export_ie_scores_20um.py
python downstream/04_interface_phenotype/train_ie_mlp_probes.py
python downstream/04_interface_phenotype/train_ie_control_baselines.py
```

`export_ie_scores_20um.py` derives an immune–epithelial (IE) enrichment score from the original mIHC images. The score measures local panCK enrichment around CD3-positive pixels relative to the patch-wide panCK abundance. It is a quantitative surrogate for immune–epithelial spatial organization and should not be interpreted as a direct cell-count, immune-infiltration, or physical cell–cell interaction measurement.

`train_ie_mlp_probes.py` evaluates frozen UR1, UR2, and fixed-coordinate Fusion representations using the same lightweight MLP probe architecture.

`train_ie_control_baselines.py` adds two controls:

- **Concat** — simple concatenation of spatially pooled UR1 and UR2 without attention-based fusion;
- **Concepts** — the three supervised DAPI/CD3/panCK abundance concept scores only.

The resulting files are consumed by the Figure 5d visualization script.

## 05_real_st_program — external ECM-associated transcriptomic signal

This workflow tests whether frozen CUBE representations encode transcriptome-associated information measured independently in the Visium HD experiment.

Run in order:

```bash
python downstream/05_real_st_program/build_fig5d_kegg_gene_lists.py
python downstream/05_real_st_program/compute_fig5d_program_scores.py
python downstream/05_real_st_program/export_visium_hd_ur1_fig5d.py
python downstream/05_real_st_program/export_visium_hd_fixed_fusion_fig5d.py
python downstream/05_real_st_program/train_fig5d_program_probes_v1.py
```

Despite the historical `fig5d` strings retained in these filenames, this workflow corresponds to **Figure 5e** in the final manuscript.

The target is the KEGG ECM–receptor interaction program (`hsa04512`). Seventeen pathway genes are shared between the CUBE gene space and the measured Visium HD dataset. Gene-wise standardization is fit using the 294 training-region patches only, and independent probes for UR1, UR2, and Fusion are evaluated on the 111 held-out test patches.

The fused representation is generated with a fixed global coordinate so that the probe does not receive absolute patch location through the fusion coordinate input.

## Visualization

`downstream/visualization/` contains plotting and source-data export scripts organized by manuscript figure.

### Figure 2

`visualization/fig2/` generates the H&E-to-mIHC benchmark, marker-resolved heatmap, paired per-patch comparisons, and representative reconstruction exports.

Typical order:

```bash
python downstream/visualization/fig2/export_fig2_per_patch_pearson.py
python downstream/visualization/fig2/export_fig2_test_reconstructions.py
python downstream/visualization/fig2/plot_fig2a_benchmark_dot.py
python downstream/visualization/fig2/plot_fig2b_marker_heatmap.py
python downstream/visualization/fig2/plot_fig2c_paired_comparison.py
```

### Figure 3

`visualization/fig3/` contains the pseudo-ST reconstruction summary, representative examples, real-ST spatial split, matched real-ST benchmark, per-gene comparisons, and whole-slide visualization code.

The visualization package includes its own [`README.md`](visualization/fig3/README.md). For the cleaned public repository, the safest convention is to run individual Figure 3 scripts from the repository root, for example:

```bash
python downstream/visualization/fig3/fig3a_pseudost_summary_v3.py
python downstream/visualization/fig3/fig3c_visium_hd_split_v3.py
python downstream/visualization/fig3/fig3d_real_st_benchmark_v3.py
python downstream/visualization/fig3/fig3e_gene_level_comparison_v3.py
```

Whole-slide DeepSpot inference is substantially slower than plotting from cached CUBE outputs and is therefore kept as a separate export step.

### Figure 4

`visualization/fig4/` generates the validation complementarity panel and the tissue-stratified held-out test analyses. See [`visualization/fig4/README.md`](visualization/fig4/README.md).

Typical commands:

```bash
python downstream/visualization/fig4/fig4a_validation_complementarity_v1.py
python downstream/visualization/fig4/fig4b_test_pearson_tissue_threshold_v1.py
python downstream/visualization/fig4/fig4c_mse_advantage_tissue_threshold_v1.py
```

### Figure 5

The final Figure 5 plotting scripts consume the outputs from `02_UMAP`, `03_spatial_cluster`, `04_interface_phenotype`, and `05_real_st_program`:

```bash
python downstream/visualization/fig5/plot_fig5a_final_v1.py
python downstream/visualization/fig5/plot_fig5b_final_v1.py
python downstream/visualization/fig5/plot_fig5c_spatial_clusters_v1.py
python downstream/visualization/fig5/plot_fig5d_ie_final_v1.py
python downstream/visualization/fig5/plot_fig5e_ecm_final_v1.py
```

## Required external and archived inputs

Downstream scripts may depend on four classes of input:

1. **public raw data** — HEMIT and the public colorectal cancer Visium HD dataset;
2. **external pretrained resources** — DeepSpot and its morphology/expression-model dependencies;
3. **CUBE checkpoints and exported features** — UR1, UR2, fusion, ablation, and real-ST decoder checkpoints;
4. **archived source-data/intermediate files** — fixed analysis tables used directly by the final plotting scripts.

Large artifacts are intentionally not committed to Git. They should be restored from the accompanying release archive/Zenodo record once available, or regenerated from the documented upstream steps when a complete regeneration path is present.

## Reproducibility notes

- Keep the original HEMIT train/validation/test membership unchanged.
- Use the same 256-gene order and training-derived pseudo-ST normalization statistics across all splits.
- Keep the H&E→UR2 encoder frozen during the real-ST decoder-adaptation experiments.
- Use identical spatial test regions when comparing DeepSpot, CUBE zero-shot, and decoder-adapted variants.
- Treat whole-slide maps containing training/validation regions as qualitative visualizations only; quantitative real-ST metrics are computed on the predefined held-out test region.
- Do not interpret UMAP clusters as pathologist-annotated tissue classes.
- Do not interpret the IE score as a direct measurement of immune infiltration or physical cell–cell interaction.
- When exact figure reproduction depends on an archived intermediate table, use the archived table associated with the released code version and record its checksum.
