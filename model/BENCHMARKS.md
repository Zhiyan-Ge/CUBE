# HE–mIHC benchmarks and architectural ablations

This document describes the comparison experiments for H&E-to-mIHC reconstruction. All supplied test results use the same 945 preprocessed CUBE test patches and the same physical mIHC channel order: DAPI, CD3, and panCK.

The final CUBE branch is `he_mihc/` with a marker-specific H&E-to-mIHC decoder. The other models in this document are controls or adapted external benchmarks.

## Evaluation metric

For each test patch and marker, the prediction and target images are flattened and Pearson correlation is computed over spatial pixels. A constant target or prediction is assigned correlation 0 by the supplied benchmark evaluators. The reported marker value is the mean per-sample correlation across the test set. The three-marker average is the unweighted mean of DAPI, CD3, and panCK for each sample, subsequently averaged across samples.

## Supplied test results

| Model | Checkpoint | DAPI r | CD3 r | panCK r | Three-marker average r |
|---|---|---:|---:|---:|---:|
| U-Net | `best_pearson.pt` | 0.6430 | 0.0057 | 0.9418 | 0.5302 |
| ResNet | `best_pearson.pt` | 0.7141 | 0.0000 | 0.9545 | 0.5562 |
| pix2pix U-Net | `best_pearson.pt` | 0.5048 | 0.2559 | 0.8730 | 0.5446 |
| pix2pix ResNet | `best_pearson.pt` | 0.6266 | 0.5523 | 0.9462 | 0.7084 |
| CUBE shared decoder | `best_pearson.pt` | 0.6809 | 0.5397 | 0.8745 | 0.6984 |
| CUBE marker-specific decoder | `best_pearson.pt` | 0.7105 | 0.5356 | 0.9308 | 0.7256 |
| HEMIT-512 adapted | `epoch_80.pt` | 0.7049 | 0.5328 | 0.9552 | 0.7310 |

The marker-specific CUBE branch improves the three-marker average over its shared-decoder ablation and over the standard CNN/pix2pix comparisons. Its average is close to, but slightly below, the supplied HEMIT-512 adapted result. These experiments do not support a claim that CUBE universally outperforms HEMIT or establishes state of the art.

## Standard CNN and pix2pix comparisons

The `baselines/` package supports four settings selected by `BENCHMARK` in `baselines/config.py`:

| `BENCHMARK` | Generator | Training objective |
|---|---|---|
| `unet` | U-Net | L1 only |
| `resnet` | ResNet generator | L1 only |
| `pix2pix_unet` | U-Net | least-squares GAN + 30 × L1 |
| `pix2pix_resnet` | ResNet generator | least-squares GAN + 30 × L1 |

The same dataset wrapper reads `he_matrix_512` and `mihc_matrix_512` and maps `[0, 1]` to `[-1, 1]` to match the Tanh-based generators.

Released defaults are batch size 2, 50 epochs, Adam with learning rate `3e-5` and beta1 0.5, step decay at epoch 50, and AMP enabled. `best_pearson.pt` is selected by maximum validation three-marker average Pearson.

Configure:

```python
# baselines/config.py
BENCHMARK = "unet"  # unet, resnet, pix2pix_unet, or pix2pix_resnet
TRAIN_DIR = "/path/to/train/pkl"
VAL_DIR = "/path/to/val/pkl"
TEST_DIR = "/path/to/test/pkl"
OUTPUT_ROOT = "/path/to/baseline/output"
GPU_ID = 0
EVAL_SPLIT = "test"
```

Run one model at a time from `model/`:

```bash
python -m baselines.trainer
python -m baselines.evaluate
```

The trainer writes each experiment under `OUTPUT_ROOT/<BENCHMARK>/`. Repeat after changing `BENCHMARK` for all four comparisons.

The GAN code in `baselines/` is intentional for the pix2pix comparisons. It is independent of the final CUBE HE–mIHC branch, which contains no GAN training or discriminator checkpoint.

## HEMIT-512 adapted benchmark

`hemit512/` ports the supplied HEMIT architecture and training protocol to the CUBE pickle pipeline. This is an **adapted 512 × 512 benchmark**, not a byte-for-byte reproduction of the original HEMIT training setup.

The documented adaptations include:

- image size changed from 1,024 to 512;
- feature-matching top-k changed from 1,000 to 250 to preserve density after halving both spatial axes;
- CUBE's paired preprocessed train/validation/test splits and channel order;
- 80 total epochs (50 fixed-rate plus 30 after the step in the supplied protocol); and
- FP32 training because AMP was disabled for numerical stability in this attention-heavy model.

Other recorded settings include batch size 2, learning rate `3e-5`, LSGAN, L1 weight 30, patch size 32, window size 64, embedding dimension 96, and depths `(2, 2, 6, 2)`.

Configure `hemit512/config.py`, then run the checks and training from `model/`:

```bash
python -m hemit512.smoke_test
python -m hemit512.preflight
python -m hemit512.trainer
python -m hemit512.evaluate
```

`smoke_test.py` checks a complete synthetic generator/discriminator update. `preflight.py` runs ten real training steps and a validation inference check. The evaluator currently loads `epoch_80.pt` explicitly and writes `test_pearson.json` when `EVAL_SPLIT = "test"`. Keep the reported checkpoint and JSON together.

Because resolution, feature-matching density, and input pipeline were adapted, label this comparator “HEMIT-512 adapted” in figures and tables. Do not label it an exact reproduction of the original HEMIT paper.

## Shared-decoder ablation

The shared-decoder experiment is an architectural ablation of CUBE HE–mIHC. It retains separate decoder instances for each reconstruction path, but H&E-to-mIHC uses one common three-channel `ImageDecoder` rather than three marker-specific decoder modules.

The archived shared-decoder run used:

| Setting | Archived value |
|---|---:|
| epochs | 100 |
| batch size | 16 |
| learning rate | `1e-4` |
| weight decay | `1e-5` |
| GPUs | `[2, 3]` |
| best validation Pearson epoch | 23 |

The supplied test result uses `best_pearson.pt` and 945 test patches.

## Reproducible reporting checklist

For each benchmark result retained in the paper, archive:

- the exact checkpoint named in the result JSON;
- the corresponding `training_report.json`;
- the test summary JSON;
- the frozen config values and test split;
- the code version or repository commit; and
- for externally derived models, the source version and a clear list of adaptations.

Training reports are useful provenance but do not need to be loaded by the public inference code. The checkpoint, evaluated split, metric implementation, and result JSON are the essential reproducibility chain.
