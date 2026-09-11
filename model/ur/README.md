# Unified Representation branch

This directory contains the final CUBE fusion model. It combines the H&E-derived representations produced by the HE–mIHC and HE–pseudo-ST branches and predicts three continuous mIHC concept scores.

## Required inputs

For every sample and split, the UR dataset requires:

| Source | Required content |
|---|---|
| preprocessed pickle | `normalized_mega_coord`, `concept_scores`, and the matching sample name |
| UR1 NumPy file | `float32 [256, 32, 32]` from `he_mihc/export_ur1.py` |
| UR2 NumPy file | `float32 [256, 16, 16]` from `he_st` |

The filenames without extensions must match exactly across the pickle, UR1, and UR2 directories. `URDataset` verifies the complete sample-name sets and the first feature shapes before training or evaluation.

The target order is DAPI, CD3, and panCK. These targets are Otsu-positive pixel fractions computed from the paired full-resolution mIHC patch during preprocessing.

## Architecture

The fusion model performs the following operations:

1. converts UR1 and UR2 feature maps into token sequences;
2. prepends an independently learned summary token to each sequence;
3. adds learned spatial position embeddings and a frequency encoding of the normalized global patch coordinate;
4. applies self-attention and feed-forward layers independently to UR1 and UR2;
5. applies bidirectional cross-attention (UR1 queries UR2 and UR2 queries UR1);
6. extracts both enhanced summary tokens and concatenates them into a `[512]` vector; and
7. predicts three concept scores through an MLP with a final sigmoid.

The released configuration uses feature dimension 256, 8 attention heads, feed-forward dimension 512, and dropout 0.1. The training objective is mean squared error between predicted and target concept scores.

## Configure

Edit `ur/config.py` with the split-specific directories:

```python
TRAIN_PKL_DIR = "/path/to/train/pkl"
TRAIN_UR1_DIR = "/path/to/he_mihc/output/ur1_he/train"
TRAIN_UR2_DIR = "/path/to/he_st/output/ur2_he/train"

VAL_PKL_DIR = "/path/to/val/pkl"
VAL_UR1_DIR = "/path/to/he_mihc/output/ur1_he/val"
VAL_UR2_DIR = "/path/to/he_st/output/ur2_he/val"

TEST_PKL_DIR = "/path/to/test/pkl"
TEST_UR1_DIR = "/path/to/he_mihc/output/ur1_he/test"
TEST_UR2_DIR = "/path/to/he_st/output/ur2_he/test"

OUTPUT_DIR = "/path/to/ur/output"
GPU_IDS = [0]

EVAL_CHECKPOINT = OUTPUT_DIR + "/best.pt"
EVAL_SPLIT = "test"
```

The released defaults are 80 epochs, batch size 4, AdamW, learning rate `1e-4`, weight decay `1e-5`, AMP enabled, and seed 2026.

## Train and evaluate the fusion model

Run from `model/`:

```bash
python -m ur.trainer
python -m ur.evaluate
```

`trainer.py` selects `best.pt` by minimum validation concept MSE and also writes periodic checkpoints, `last.pt`, `training_report.json`, and `concept_loss.png`. The supplied fusion run selected epoch 75.

`evaluate.py` writes:

```text
OUTPUT_DIR/evaluation/<checkpoint>_<split>/
├── ur_evaluation.json
└── predictions.csv
```

The JSON reports overall MSE/MAE/RMSE, a training-mean baseline, and per-concept MSE, Pearson correlation, R², target/prediction moments, and relative baseline improvement. `predictions.csv` is required by the tissue-stratified analysis.

## Ablations

`ur1_ablation/` and `ur2_ablation/` contain architecture-matched single-representation controls:

- UR1-only uses the `[256, 32, 32]` representation plus the global coordinate;
- UR2-only uses the `[256, 16, 16]` representation plus the global coordinate.

Each control uses two self-attention/feed-forward stages and the same type of concept head, but no cross-modal fusion.

Configure the corresponding paths in each ablation `config.py`, then run:

```bash
python -m ur1_ablation.trainer
python -m ur1_ablation.evaluate

python -m ur2_ablation.trainer
python -m ur2_ablation.evaluate
```

The supplied ablation configs default to `EVAL_SPLIT = "val"`. Evaluate both validation and test explicitly by changing only `EVAL_SPLIT` between runs and retaining the same frozen `best.pt`. The supplied best epochs are 66 for UR1-only and 46 for UR2-only.

## Supplied validation results

The validation set contains 630 patches and was the checkpoint-selection domain.

| Model | Overall MSE | Improvement over train-mean baseline | DAPI r | CD3 r | panCK r |
|---|---:|---:|---:|---:|---:|
| UR1 + UR2 fusion | 0.006609 | 75.19% | 0.4520 | 0.8829 | 0.9362 |
| UR1-only | 0.008823 | 66.87% | 0.3184 | 0.8787 | 0.9217 |
| UR2-only | 0.008743 | 67.17% | 0.3685 | 0.7662 | 0.9098 |

On this split, fusion has the lowest overall MSE.

## Supplied test results and tissue sensitivity

The full test set contains 945 patches:

| Model | Overall MSE | Improvement over train-mean baseline | DAPI r | CD3 r | panCK r |
|---|---:|---:|---:|---:|---:|
| UR1 + UR2 fusion | 0.005571 | 57.68% | 0.9235 | 0.8132 | 0.6634 |
| UR1-only | 0.003690 | 71.97% | 0.9187 | 0.8186 | 0.8066 |
| UR2-only | 0.006053 | 54.02% | 0.8769 | 0.6585 | 0.6801 |

The full-test ranking reverses relative to validation: UR1-only has lower MSE than fusion. Static QC found a small group of near-blank test patches with very large errors, especially for fusion. This result must be reported rather than hidden.

The original four-threshold diagnostic analysis shows the following overall MSE after retaining patches at or above each H&E tissue fraction:

| Retained subset | N | Fusion | UR1-only | UR2-only | Fusion improvement vs UR1 | Fusion improvement vs UR2 |
|---|---:|---:|---:|---:|---:|---:|
| Tissue ≥ 1% | 924 | 0.001929 | 0.001899 | 0.002495 | −1.54% | 22.71% |
| Tissue ≥ 5% | 917 | 0.001922 | 0.001876 | 0.002511 | −2.45% | 23.46% |
| Tissue ≥ 10% | 911 | 0.001933 | 0.001888 | 0.002527 | −2.43% | 23.48% |
| Tissue ≥ 25% | 892 | 0.001577 | 0.001918 | 0.002565 | 17.78% | 38.52% |

These tissue thresholds are post hoc diagnostic strata applied uniformly to the three frozen models; they must not be presented as thresholds optimized on the test set. The final Figure 4 sensitivity series additionally includes 30%, 50%, and 75% retained-tissue thresholds via `downstream/visualization/fig4/tissue_stratified_evaluation_extended.py`. The complete reported threshold series is retained rather than selecting a single cutoff according to test performance. Tissue fraction is defined by near-white pixels rather than a histological tissue-segmentation model. See the [analysis documentation](../analysis/README.md) for definitions and output files.

## Interpretation

The validation and tissue-stratified results support complementary information in UR1 and UR2, but the unfiltered test result shows that fusion is not uniformly superior under severe distribution shift or near-blank inputs. Report validation, full-test, and tissue-stratified findings together. Do not summarize this experiment as “fusion always outperforms both ablations.”
