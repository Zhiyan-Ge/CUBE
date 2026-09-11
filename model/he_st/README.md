# HE–pseudo-ST branch

This directory contains the CUBE branch that learns UR2 by bridging H&E patches and a frozen DeepSpot-derived spatial pseudo-transcriptomic target.

The target in this branch is **not independently measured spatial transcriptomics**. It is a pseudo-ST matrix predicted from H&E during preprocessing. The reported metrics quantify how well the branch reconstructs that frozen pseudo-ST representation.

## Inputs and normalization

`HESTDataset` reads:

| Input | Shape | Processing |
|---|---:|---|
| H&E | `[3, 256, 256]` | loaded from `he_matrix_256` in `[0, 1]` |
| pseudo-ST | `[16, 16, 256]` | loaded from `st_matrix` and normalized gene-wise |

Pseudo-ST is normalized using the training-set statistics in `ST_STATS_PATH`:

```python
st = (st - mean) / safe_std
```

Both `mean` and `safe_std` contain 256 entries and are reshaped to `[1, 1, 256]`. The archived `st_stats.npz` was computed from the 3,717 training files only, corresponding to 951,552 spatial positions. The statistics file and all pseudo-ST matrices must use exactly the same gene order.

Use the supplied archived `st_stats.npz` for exact reproduction of the released experiments. If the statistics file needs to be regenerated, run `data_preprocessing/generate_st_stats.py` on the frozen training pickle directory only. Do not recompute the statistics from validation or test data.

## Architecture

The branch preserves two named submodels for checkpoint compatibility:

- `model3`: H&E → H&E reconstruction, pseudo-ST reconstruction, and H&E-derived UR2;
- `model4`: pseudo-ST → H&E reconstruction, pseudo-ST reconstruction, and pseudo-ST-derived UR2.

The H&E encoder maps `[3, 256, 256]` to UR2 `[256, 16, 16]`. The ST encoder first reorders `[16, 16, 256]` to channel-first format and retains the same 16 × 16 spatial grid. Independent H&E and ST decoders are used in both submodels. Do not rename `model3`, `model4`, or their child modules when loading existing checkpoints.

## Objective and defaults

The branch objective includes:

- H&E self-reconstruction: L1 + SSIM;
- H&E→ST and ST→ST reconstruction: mean squared error in normalized pseudo-ST space;
- ST→H&E auxiliary reconstruction; and
- weak MSE alignment between pooled H&E-derived and ST-derived UR2.

No Sinkhorn loss or adversarial loss is used by this branch.

The released defaults in `config.py` are:

| Setting | Value |
|---|---:|
| epochs | 100 |
| batch size | 4 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-5` |
| AMP | enabled |
| seed | 2026 |
| HE→ST weight (`BETA3`) | 1.0 |
| ST→HE auxiliary weight (`BETA4`) | 0.1 |
| UR2 alignment weight (`GAMMA2`) | 0.05 |

## Configure

Edit `config.py`:

```python
TRAIN_DIR = "/path/to/train/pkl"
VAL_DIR = "/path/to/val/pkl"
TEST_DIR = "/path/to/test/pkl"
ST_STATS_PATH = "/path/to/st_stats.npz"
OUTPUT_DIR = "/path/to/he_st/output"
GPU_IDS = [0]

EVAL_CHECKPOINT = OUTPUT_DIR + "/best.pt"
EVAL_SPLIT = "test"
```

The paper evaluation uses the test split (`N = 945`). Validation is used for checkpoint selection.

## Train

Run from `model/`:

```bash
python -m he_st.trainer
```

The trainer selects `best.pt` by the minimum validation H&E→ST MSE and also writes `epoch_N.pt`, `last.pt`, `training_report.json`, `total_loss.png`, and `loss_components.png`.

With `SAVE_UR2 = True`, training reloads `best.pt` and exports train and validation UR2 arrays under `OUTPUT_DIR/ur2_he/`. The supplied training report identifies epoch 31 as the minimum validation H&E→ST MSE (`0.181558`).

## Evaluate

```bash
python -m he_st.evaluate_st
```

The evaluator writes:

```text
OUTPUT_DIR/evaluation/<checkpoint>_<split>/st_evaluation.json
```

It reports H&E→ST and ST→ST MSE, zero-prediction baseline MSE, global Pearson correlation, sample-wise Pearson, gene-wise global Pearson, cross-sample gene-abundance Pearson, and within-patch spatial Pearson. Constant or invalid vectors are excluded from the corresponding Pearson summary and counted as invalid.

The frozen final checkpoint used for the manuscript (`test13/he_st/best.pt`) gives the following H&E→ST results on the 945 held-out test patches:

| Metric | Value |
|---|---:|
| samples | 945 |
| MSE | 0.311899 |
| global Pearson r | 0.802474 |
| sample-wise Pearson r, mean | 0.769287 |
| gene-wise global Pearson r, mean | 0.795083 |
| cross-sample abundance Pearson r, mean | 0.889211 |
| within-patch spatial Pearson r, mean | 0.723359 |

These are the frozen test values used by the final manuscript and Figure 3. Additional diagnostic statistics should be regenerated from the evaluation outputs of this same checkpoint rather than copied from earlier experimental runs.

## Spatial diagnostics

Run after setting the same evaluation checkpoint and split:

```bash
python -m he_st.diagnose_st
```

This writes gene-level and sample-level CSV files, three Pearson histograms, a target-versus-prediction standard-deviation plot, and `spatial_diagnostics_summary.json` to the same evaluation directory. It uses the same metric decomposition as `evaluate_st.py` and does not retrain the model.

## Visualization

For reproducible qualitative examples:

```bash
python -m he_st.diagnose_st
python -m he_st.visualize_st
```

`visualize_st.py` uses the gene diagnostic CSV to select a mixture of high-variance, best-spatial, median-spatial, and worst-spatial genes. It deterministically selects `EVAL_NUM_SAMPLES` samples with `SEED` and writes target, prediction, and residual heatmaps under:

```text
OUTPUT_DIR/evaluation/<checkpoint>_<split>/st_visualizations/
```

Run `diagnose_st.py` first so that gene selection uses the saved diagnostic statistics.

## Export H&E-derived UR2

After training, run:

```bash
python -m he_st.export_ur2
```

The standalone exporter loads `model3.encoder` from `best.pt` and writes test-set `float32 [256, 16, 16]` arrays to:

```text
OUTPUT_DIR/ur2_he/test/<sample_name>.npy
```

The standalone script exports **test only**. Train and validation UR2 arrays are exported by `trainer.py` when `SAVE_UR2 = True`; run both stages to prepare all three splits for the UR branch.
