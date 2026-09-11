# H&E tissue-quality analysis

This directory contains two post hoc diagnostic analyses for the CUBE UR experiments:

1. `tissue_qc.py` quantifies the amount of non-near-white H&E content in every train, validation, and test patch; and
2. `tissue_stratified_evaluation.py` recalculates frozen fusion, UR1-only, and UR2-only results within common tissue-fraction strata.

These scripts do not train or modify any model.

## Tissue-fraction definition

The analysis defines a pixel as near-white background when all three H&E channels are at least 245 on the `[0, 255]` scale:

```text
near_white = (R >= 245) and (G >= 245) and (B >= 245)
tissue_fraction = 1 - mean(near_white)
```

This is a reproducible image-content heuristic. It is not a stain-deconvolution procedure, a tissue-segmentation model, or a direct measurement of viable tumor area. Accordingly, the README uses “non-near-white H&E fraction” and “tissue fraction” only in this operational sense.

The core diagnostic script `tissue_stratified_evaluation.py` evaluates 1%, 5%, 10%, and 25% tissue-fraction thresholds. The final Figure 4 sensitivity analysis extends the retained-tissue series with 30%, 50%, and 75% thresholds in `downstream/visualization/fig4/tissue_stratified_evaluation_extended.py`. These are post hoc sensitivity strata applied consistently to the frozen models; the reported series is retained rather than selecting a cutoff according to test performance.

## Configure and run tissue QC

`tissue_qc.py` imports the train/validation/test pickle paths and `OUTPUT_DIR` from `ur/config.py`. Configure those paths first, then run from `model/`:

```bash
python -m analysis.tissue_qc
```

For each pickle, the script finds the largest H&E-like image array, converts `[0, 1]` values back to `[0, 255]` when needed, and calculates exact-white, near-white, and tissue fractions.

Outputs are written to `ur.config.OUTPUT_DIR/tissue_qc/`:

| Output | Contents |
|---|---|
| `tissue_fraction_per_patch.csv` | split, sample name, selected H&E key, and per-patch fractions |
| `tissue_fraction_summary.csv` | split-level counts, moments, percentiles, and low-tissue counts |
| `lowest_tissue_100.csv` | the 100 lowest-tissue patches across all splits |
| `tissue_fraction_distribution.png` | full split distributions |
| `low_tissue_distribution.png` | 0–30% tissue region |

If SciPy is installed, the script also prints validation-versus-test Kolmogorov–Smirnov and Fisher exact tests. These console tests are descriptive diagnostics and are not model-selection criteria.

## Supplied QC summary

| Split | N | Mean tissue fraction | Median | <1% | <5% | <10% | <25% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,717 | 0.8698 | 0.9414 | 4 (0.11%) | 9 (0.24%) | 14 (0.38%) | 34 (0.91%) |
| Validation | 630 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 |
| Test | 945 | 0.8293 | 0.9371 | 21 (2.22%) | 28 (2.96%) | 34 (3.60%) | 53 (5.61%) |

The validation patches are almost entirely non-near-white under this heuristic, whereas the test set includes a small but influential near-blank tail. This distribution difference motivated the tissue-stratified evaluation.

## Prepare tissue-stratified evaluation

First evaluate all three frozen UR models on the test split so that the following prediction files exist:

```text
<fusion output>/evaluation/best_test/predictions.csv
<UR1 output>/evaluation/best_test/predictions.csv
<UR2 output>/evaluation/best_test/predictions.csv
```

Then edit the path constants at the top of `tissue_stratified_evaluation.py`:

```python
TISSUE_CSV = "/path/to/fusion/output/tissue_qc/tissue_fraction_per_patch.csv"
FUSION_PRED_CSV = "/path/to/fusion/output/evaluation/best_test/predictions.csv"
UR1_PRED_CSV = "/path/to/ur1/output/evaluation/best_test/predictions.csv"
UR2_PRED_CSV = "/path/to/ur2/output/evaluation/best_test/predictions.csv"
OUTPUT_DIR = "/path/to/fusion/output/tissue_stratified_evaluation"
```

The current source already derives the tissue and fusion paths from `ur.config.OUTPUT_DIR`, but the UR1 and UR2 paths remain explicit placeholders and must be replaced.

## Run tissue-stratified evaluation

```bash
python -m analysis.tissue_stratified_evaluation
```

Before calculating metrics, the script enforces:

- identical sample-name sets between tissue QC and each model;
- one-to-one sample alignment;
- identical ordering after alignment; and
- numerically identical concept targets across the three prediction CSV files.

It evaluates the full test set, every low-tissue subset (`Tissue<threshold`), and every retained subset (`Tissue>=threshold`). The same sample stratum is used for all three frozen models.

Outputs are:

| Output | Contents |
|---|---|
| `tissue_stratified_metrics.csv` | overall and per-concept metrics for every model/stratum |
| `fusion_vs_ablation_by_tissue.csv` | overall MSE and fusion improvement relative to both ablations |
| `per_patch_errors.csv` | aligned per-patch absolute errors by model and concept |

## Supplied stratified results

Overall MSE from the supplied test predictions is:

| Subset | N | Fusion | UR1-only | UR2-only |
|---|---:|---:|---:|---:|
| Full | 945 | 0.005571 | 0.003690 | 0.006053 |
| Tissue < 1% | 21 | 0.165836 | 0.082502 | 0.162606 |
| Tissue ≥ 1% | 924 | 0.001929 | 0.001899 | 0.002495 |
| Tissue < 5% | 28 | 0.125065 | 0.063106 | 0.122049 |
| Tissue ≥ 5% | 917 | 0.001922 | 0.001876 | 0.002511 |
| Tissue < 10% | 34 | 0.103034 | 0.051998 | 0.100544 |
| Tissue ≥ 10% | 911 | 0.001933 | 0.001888 | 0.002527 |
| Tissue < 25% | 53 | 0.072790 | 0.033519 | 0.064761 |
| Tissue ≥ 25% | 892 | 0.001577 | 0.001918 | 0.002565 |

For the `Tissue ≥ 25%` subset, fusion reduces MSE by 17.78% relative to UR1-only and 38.52% relative to UR2-only. For the full test set and the lower retained thresholds, UR1-only remains slightly better than fusion. Both facts should be reported.

## Reporting guidance

- Treat the full test set as the primary unfiltered evaluation.
- Present tissue-stratified results as sensitivity analyses that explain the observed distribution shift.
- State the near-white heuristic and every reported threshold.
- Do not discard low-tissue patches silently or choose a threshold because it maximizes fusion performance.
- Do not infer that near-white background is the only cause of every error; this analysis establishes an association between the heuristic and error magnitude.
