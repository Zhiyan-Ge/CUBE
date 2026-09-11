# HE–mIHC branch

This directory contains the final CUBE branch that learns UR1 by bridging paired H&E and multiplex immunohistochemistry (mIHC) patches. The released model uses a marker-specific decoder for H&E-to-mIHC prediction and does not use adversarial training.

## Inputs and outputs

`HEMIHCDataset` reads the following fields from each preprocessed pickle:

| Field | Shape | Meaning |
|---|---:|---|
| `he_matrix_512` | `[3, 512, 512]` | H&E image in `[0, 1]` |
| `mihc_matrix_512` | `[3, 512, 512]` | mIHC channels DAPI, CD3, and panCK in `[0, 1]` |
| `sample_name` | scalar string | sample identifier |

The HE encoder produces H&E-derived UR1 with shape `[256, 32, 32]` for each patch.

## Architecture

The complete branch contains two named submodels whose attribute names are preserved for checkpoint compatibility:

- `model1`: H&E input → H&E reconstruction, marker-specific mIHC reconstruction, and H&E-derived UR1;
- `model2`: mIHC input → H&E reconstruction, mIHC reconstruction, and mIHC-derived UR1.

Both encoders are convolutional residual encoders. The final H&E-to-mIHC path uses independent DAPI, CD3, and panCK decoder modules. The other image decoders retain the shared `ImageDecoder` class. Do not rename `model1`, `model2`, or their decoder attributes when using the supplied checkpoints.

The historical shared-decoder architecture is retained in `../he_mihc_shared_decoder/` as an ablation and is discussed in [BENCHMARKS.md](../BENCHMARKS.md#shared-decoder-ablation).

## Objective and frozen defaults

The training objective combines self-reconstruction, cross-modal reconstruction, and weak alignment of the two UR1 representations. The final HE-to-mIHC loss also includes multi-scale Pearson supervision, marker weighting, foreground-aware weighting, and a CD3 soft-Dice term as implemented in `loss/recon_loss.py`, `loss/alignment_loss.py`, and `loss/model_loss.py`.

The released defaults in `config.py` are:

| Setting | Value |
|---|---:|
| epochs | 60 |
| per-GPU batch size | 4 |
| optimizer | AdamW |
| learning rate | `5e-5` |
| weight decay | `1e-5` |
| AMP | enabled |
| seed | 2026 |
| group parameter | 8 |
| Pearson weight | 0.75 |
| Pearson scales | 1, 2, 4 |
| scale weights | 0.5, 0.3, 0.2 |
| CD3 soft-Dice weight | 0.2 |
| mIHC foreground threshold | 0.05 |
| foreground extra weight | 2.0 |
| DAPI/CD3/panCK channel weights | 1.0 / 2.0 / 1.0 |
| paired spatial augmentation | enabled |
| UR1 alignment weight | 0.05 |

`BATCH_SIZE` is per GPU under distributed data parallelism; the global batch size is `BATCH_SIZE × len(GPU_IDS)`.

## Configure

Edit only the paths and hardware settings needed for your system in `config.py`:

```python
TRAIN_DIR = "/path/to/train/pkl"
VAL_DIR = "/path/to/val/pkl"
TEST_DIR = "/path/to/test/pkl"
OUTPUT_DIR = "/path/to/he_mihc/output"
GPU_IDS = [0]
```

The archived experiment used 3,717 training, 630 validation, and 945 test patches. Keep `TRAIN_SAMPLES` and `VAL_SAMPLES` as `None` to use every sample.

## Train

Run from the parent `model/` directory:

```bash
python -m he_mihc.trainer
```

Training writes:

- `best_pearson.pt`: maximum validation mean Pearson across DAPI, CD3, and panCK;
- `best_cross.pt`: minimum validation HE→mIHC plus mIHC→HE loss;
- `best.pt`: an alias saved with `best_cross.pt`;
- `best_total.pt`: minimum full validation objective;
- `epoch_N.pt`, `last.pt`, loss plots, and `training_report.json`.

Each checkpoint contains the epoch, model state, and optimizer state. GAN/discriminator states are not part of the final branch.

In the supplied final experiment, both the best validation Pearson and best cross-modal criterion occurred at epoch 24. The paper HE→mIHC Pearson result uses `best_pearson.pt`.

## Test Pearson evaluation

```bash
python -m he_mihc.evaluate_pearson
```

The script evaluates `best_pearson.pt` when present and reads `TEST_DIR` directly. For each sample and channel, it flattens the spatial pixels and computes Pearson correlation. The reported channel result is the mean of these per-sample correlations; `average` is the unweighted mean of the three channel correlations for each sample, subsequently averaged across samples.

Outputs are written under:

```text
OUTPUT_DIR/diagnostics/pearson/
├── best_pearson_pearson.csv
├── best_pearson_pearson_summary.json
└── pearson_distribution.png
```

Supplied test-set results (`N = 945`) are:

| Metric | Mean Pearson r | Median | Standard deviation |
|---|---:|---:|---:|
| DAPI | 0.7105 | 0.8096 | 0.2448 |
| CD3 | 0.5356 | 0.5609 | 0.1320 |
| panCK | 0.9308 | 0.9449 | 0.1097 |
| Three-marker average | 0.7256 | 0.7714 | 0.1252 |

These values are transcribed from `best_pearson_pearson_summary.json`; they are not recomputed by this documentation.

## Reconstruction diagnostics

```bash
python -m he_mihc.evaluate_recon
python -m he_mihc.visualize_recon
```

`evaluate_recon.py` is a validation-set diagnostic. It calculates training-set mean baselines, L1, SSIM, per-channel errors, and foreground/background errors for the available `epoch_20.pt`, `epoch_30.pt`, `best.pt`, and `last.pt` checkpoints, then saves `OUTPUT_DIR/diagnostics/diagnostic_metrics.json`.

`visualize_recon.py` selects eight validation samples deterministically using `SEED` and compares all available checkpoints listed in the script. It writes reconstruction overviews and H&E-to-mIHC channel panels to `OUTPUT_DIR/diagnostics/reconstruction/`.

## Export H&E-derived UR1

After freezing `best_pearson.pt`, run:

```bash
python -m he_mihc.export_ur1
```

The exporter loads only `model1.encoder` from `best_pearson.pt` and writes one `float32 [256, 32, 32]` NumPy array per patch:

```text
OUTPUT_DIR/ur1_he/
├── train/<sample_name>.npy
├── val/<sample_name>.npy
└── test/<sample_name>.npy
```

The trainer can also export train/validation features after training, but the standalone exporter is the recommended release step because it explicitly uses `best_pearson.pt` and covers all three splits.

## Interpretation and comparison

The branch predicts spatial fluorescence patterns from H&E and learns a transferable latent representation; it does not measure protein expression directly from a new assay. Benchmark results, including the shared-decoder ablation and HEMIT-512 adaptation, are provided in [BENCHMARKS.md](../BENCHMARKS.md).
