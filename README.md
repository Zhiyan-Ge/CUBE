# CUBE

**CUBE (Colorectal Universal Representation & Bridge Encoder)** is a multimodal representation-learning framework for integrating histomorphology, spatial protein phenotypes, and transcriptome-associated signals from incompletely paired tissue data.

CUBE uses H&E histology as a bridge modality. It first learns two H&E-derived representations independently: **UR1** from paired H&E–mIHC data and **UR2** from H&E with DeepSpot-derived pseudo-spatial transcriptomic supervision. The two representations are then integrated through attention-based fusion and biologically grounded using mIHC-derived concept scores.

This repository contains the preprocessing, model training, benchmarking, ablation, transfer, downstream analysis, and figure-generation code used in the CUBE study.

## Resources

- **Preprint:** Ge Z, Cai H. *CUBE: Multimodal Representation Learning Reveals Biological Structure Across Histomorphology, Spatial Protein Phenotypes, and Transcriptome-Associated Signals.* bioRxiv (2026).  
  https://doi.org/10.64898/2026.09.14.751379

- **Source data:**  
  https://doi.org/10.5281/zenodo.22713021

- **Pretrained model checkpoints:**  
  https://doi.org/10.5281/zenodo.22713201

- **Source code:**  
  https://github.com/Zhiyan-Ge/CUBE

## Overview

The released workflow contains three principal model components:

1. **H&E–mIHC branch (UR1)**  
   Learns modality-aligned H&E and mIHC representations and supports bidirectional reconstruction between histology and DAPI/CD3/panCK mIHC images.

2. **H&E–pseudo-ST branch (UR2)**  
   Learns a transcriptome-associated representation using DeepSpot-derived pseudo-ST supervision. Experimentally measured Visium HD data are used separately for transfer and validation rather than as the main pretraining target.

3. **UR fusion branch**  
   Integrates H&E-derived UR1 and UR2 using self-attention, bidirectional cross-attention, and global spatial-coordinate features. The fused representation is supervised by continuous DAPI-, CD3-, and panCK-derived concept scores.

The final analyses test whether the fused representation retains biologically structured information beyond its direct training targets, including immune–epithelial spatial organization and an independently measured ECM–receptor interaction transcriptomic program.

## Repository structure

```text
CUBE/
├── data_preprocessing/        # HEMIT preprocessing and DeepSpot-derived pseudo-ST generation
├── model/                     # CUBE branches, ablations, benchmarks, and model-level analyses
├── downstream/                # real-ST transfer, latent-space and biological downstream analyses
│   ├── 01_real_st/
│   ├── 02_UMAP/
│   ├── 03_spatial_cluster/
│   ├── 04_interface_phenotype/
│   ├── 05_real_st_program/
│   └── visualization/
└── environment/               # archived reference environment
```

Large datasets, trained checkpoints, and generated results are not stored in the Git repository. The released scripts use the following repository-root-relative locations where applicable:

```text
data/       # external and processed datasets
models/     # released or locally trained checkpoints
result/     # downstream and visualization outputs
```

Detailed documentation is available in:

- [`data_preprocessing/README.md`](data_preprocessing/README.md)
- [`model/README.md`](model/README.md)
- [`model/BENCHMARKS.md`](model/BENCHMARKS.md)
- [`downstream/README.md`](downstream/README.md)

## Reference environment

The reported experiments were run with:

```text
Python        3.12.2
PyTorch       2.10.0
CUDA build    12.8
Torchvision   0.25.0
GPU           NVIDIA L40
Seed          2026
```

A snapshot of the complete shared Python environment is provided in:

```text
environment/cube_environment_snapshot.txt
```

The snapshot is intended for exact provenance rather than as a minimal dependency specification.

## Data

### HEMIT

CUBE was primarily developed using the HEMIT colorectal cancer dataset containing paired H&E and multiplex immunohistochemistry patches. The original HEMIT split is retained throughout the study:

```text
Train       3717 patches
Validation   630 patches
Test         945 patches
```

The mIHC channel order used throughout this repository is:

```text
0: DAPI
1: CD3
2: panCK
```

Raw HEMIT data are not redistributed in this repository. See [`data_preprocessing/README.md`](data_preprocessing/README.md) for preprocessing assumptions and expected input organization.

### DeepSpot-derived pseudo-ST

Pseudo-ST supervision is generated from H&E using the pretrained DeepSpot workflow. The resulting matrix for each HEMIT patch has shape:

```text
16 × 16 × 256
```

These values are model-derived pseudo-transcriptomic targets and are **not** independently measured spatial transcriptomics. DeepSpot and its required external pretrained resources must be obtained from their original sources. See [`data_preprocessing/README.md`](data_preprocessing/README.md) for the exact resources used in the study.

### Visium HD

Experimentally measured spatial transcriptomics transfer and downstream validation use a public colorectal cancer Visium HD section analyzed at the 16 μm aggregated-bin resolution. The real-ST workflow is documented under [`downstream/01_real_st/`](downstream/01_real_st/) and summarized in [`downstream/README.md`](downstream/README.md).

## Preprocessing

The preprocessing pipeline converts paired H&E/mIHC images into the serialized inputs used by all CUBE branches and generates the DeepSpot-derived pseudo-ST targets.

```bash
(cd data_preprocessing && python preprocessing.py)
```

Required dataset and DeepSpot paths can be supplied through the environment variables documented in [`data_preprocessing/README.md`](data_preprocessing/README.md).

After preprocessing, generate pseudo-ST normalization statistics **using the training split only**:

```bash
python data_preprocessing/generate_st_stats.py /path/to/train/pkl \
    --output /path/to/train/st_stats.npz
```

Validation and test samples must use the same training-derived gene order, mean, and standard deviation.

## Training CUBE

Model commands are run from the `model/` directory after editing the path placeholders in the corresponding `config.py` files.

```bash
cd model
```

### 1. Train UR1 from H&E–mIHC

```bash
python -m he_mihc.trainer
python -m he_mihc.evaluate_pearson
python -m he_mihc.export_ur1
```

The H&E-derived UR1 representation has shape:

```text
[256, 32, 32]
```

For the reported H&E-to-mIHC result, the selected checkpoint is the maximum-validation-Pearson checkpoint (`best_pearson.pt`).

### 2. Train UR2 from H&E–pseudo-ST

```bash
python -m he_st.trainer
python -m he_st.evaluate_st
python -m he_st.export_ur2
```

The H&E-derived UR2 representation has shape:

```text
[256, 16, 16]
```

The same training-set `st_stats.npz` must be used for training, validation, testing, and downstream export.

### 3. Train the fused UR model

After exporting matched UR1 and UR2 features for all splits:

```bash
python -m ur.trainer
python -m ur.evaluate
```

The fusion model produces a 512-dimensional summary representation and predicts three continuous mIHC-derived concept scores.

### 4. Train single-representation ablations

```bash
python -m ur1_ablation.trainer
python -m ur1_ablation.evaluate

python -m ur2_ablation.trainer
python -m ur2_ablation.evaluate
```

The ablation models are independently trained architecture-matched controls rather than masked versions of the trained fusion model.

More detailed architecture, hyperparameter, checkpoint, and output information is provided in [`model/README.md`](model/README.md).

## H&E–mIHC benchmarks

The repository includes:

- U-Net
- ResNet
- pix2pix U-Net
- pix2pix ResNet
- a shared-decoder CUBE ablation
- HEMIT-512 adapted

These comparisons use the same HEMIT split and evaluation pipeline as CUBE. See [`model/BENCHMARKS.md`](model/BENCHMARKS.md) for the exact adaptations, training settings, and reported metrics.

## Downstream analyses

The downstream workflow is organized by analysis rather than by model component:

| Directory | Main role | Paper output |
|---|---|---|
| `01_real_st/` | Visium HD preparation, frozen-encoder transfer, decoder adaptation, DeepSpot comparison | Figure 3 |
| `02_UMAP/` | fixed-coordinate fused-UR export and latent-space visualization | Figure 5a |
| `03_spatial_cluster/` | K-means analysis and spatial remapping of fused-UR states | Figure 5b–c |
| `04_interface_phenotype/` | immune–epithelial enrichment score and frozen-representation probes | Figure 5d |
| `05_real_st_program/` | external ECM-associated transcriptomic program probing | Figure 5e |
| `visualization/` | final plotting scripts for manuscript figures | Figures 2–5 |

See [`downstream/README.md`](downstream/README.md) for execution order and required intermediate files.

## Reproducing the paper results

There are two intended routes.

### Route A — use released checkpoints

For reproducing the reported evaluations and downstream analyses without retraining the full model:

1. download the released CUBE checkpoint archive from the model repository/Zenodo record (link to be added);
2. place checkpoints under the expected `models/` locations or update the path constants/configuration files;
3. prepare the required public datasets and preprocessing outputs;
4. follow [`downstream/README.md`](downstream/README.md); and
5. run the corresponding scripts under `downstream/visualization/`.

### Route B — train from scratch

For complete retraining:

```text
HEMIT raw data
    ↓
data_preprocessing/
    ↓
HE–mIHC branch → UR1
    ↓
HE–pseudo-ST branch → UR2
    ↓
UR fusion + single-UR ablations
    ↓
real-ST and biological downstream analyses
    ↓
figure-generation scripts
```

Use the frozen split sizes and checkpoint-selection criteria described in the branch documentation when attempting exact reproduction.

## Pretrained checkpoints and archived artifacts

The Git repository is intended to contain source code only. Large trained checkpoints and research artifacts will be archived separately with a persistent DOI.

The release archive should contain, at minimum, the checkpoints used for the manuscript analyses together with a manifest describing their role and corresponding code version. External pretrained models such as DeepSpot/UNI are not redistributed by CUBE and must be obtained from their original sources.

Zenodo / model archive DOI: **to be added**.

## Notes on paths

Machine-specific absolute paths have been removed from the public code. Most downstream scripts use paths relative to the repository root, for example:

```text
./data/...
./models/...
./result/...
```

Run downstream scripts from the repository root unless a subdirectory README explicitly states otherwise. Scripts that retain experiment-specific path constants are intentionally lightweight; edit those constants to match a different local layout rather than changing the scientific settings.

## Citation

If you use CUBE, please cite the accompanying manuscript. Citation information will be updated after the preprint or journal record is available.

```text
CUBE: Multimodal Representation Learning Reveals Biological Structure Across
Histomorphology, Spatial Protein Phenotypes, and Transcriptome-Associated Signals.
```

## License

Original CUBE source code is released under the [MIT License](LICENSE), except for third-party components identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The CUBE license applies only to project-authored source code. Third-party code, datasets, pathway annotations, pretrained models, model weights, and other external resources remain subject to their original licenses and terms of use and are not relicensed by CUBE.
