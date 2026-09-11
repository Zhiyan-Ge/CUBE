# CUBE models

This directory contains the model-training, evaluation, ablation, benchmarking, and image-quality analysis code for CUBE (Colorectal Universal Representation & Bridge Encoder).

CUBE learns two H&E-derived representations:

- **UR1**, learned by bridging H&E and mIHC spatial protein phenotypes; and
- **UR2**, learned by bridging H&E and a DeepSpot-derived spatial pseudo-transcriptomic program.

The UR branch then fuses H&E-derived UR1 and UR2 with self-attention, bidirectional cross-attention, and global patch coordinates. A three-output concept head predicts the DAPI, CD3, and panCK concept scores used as the supervised readout.

## Directory guide

| Path | Role | Documentation |
|---|---|---|
| `he_mihc/` | final marker-specific HE–mIHC branch and UR1 export | [HE–mIHC README](he_mihc/README.md) |
| `he_st/` | HE–pseudo-ST branch and UR2 export | [HE–ST README](he_st/README.md) |
| `ur/` | final UR1/UR2 fusion model | [UR README](ur/README.md) |
| `ur1_ablation/` | architecture-matched UR1-only ablation | [UR README](ur/README.md#ablations) |
| `ur2_ablation/` | architecture-matched UR2-only ablation | [UR README](ur/README.md#ablations) |
| `analysis/` | H&E tissue QC and tissue-stratified evaluation | [Analysis README](analysis/README.md) |
| `he_mihc_shared_decoder/` | shared-decoder architectural ablation | [Benchmark notes](BENCHMARKS.md#shared-decoder-ablation) |
| `baselines/` | U-Net, ResNet, pix2pix U-Net, and pix2pix ResNet comparisons | [Benchmark notes](BENCHMARKS.md) |
| `hemit512/` | HEMIT adapted to 512 × 512 CUBE inputs | [Benchmark notes](BENCHMARKS.md) |
| `loss/` | reconstruction, alignment, branch, and concept losses | used by the model branches |

The final CUBE model is the combination of `he_mihc/`, `he_st/`, and `ur/`. The shared-decoder, single-UR, baseline, and HEMIT directories are comparison experiments rather than alternative release defaults.

## Execution convention

Run model commands from this `model/` directory. Several scripts use package imports such as `from he_mihc...`, `from ur...`, or `from loss...` and therefore assume that `model/` is on `PYTHONPATH`:

```bash
cd model
python -m he_mihc.trainer
```

Each branch keeps its experiment paths and hyperparameters in `config.py`. Replace every `path/to/...` placeholder before running. The cleanup deliberately retains the original lightweight Python configuration style; no new configuration framework was introduced.

## Reference environment

The study was run in a shared Conda base environment rather than a project-specific virtual environment. The recorded and GPU-tested reference environment was:

| Component | Recorded version |
|---|---|
| Python | 3.12.2 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0 |
| PyTorch CUDA build | 12.8 |
| cuDNN | 9.10.2 (`91002`) |
| NVIDIA driver | 550.127.08 |
| `nvidia-smi` reported CUDA | 12.4 |
| GPU | NVIDIA L40, 46,068 MiB |

An actual CUDA allocation was tested successfully with `CUDA_VISIBLE_DEVICES=3`. The PyTorch CUDA build version and the maximum CUDA version displayed by `nvidia-smi` describe different layers of the stack and are therefore not expected to be identical.

Selected recorded Python packages are:

```text
anndata==0.12.18
h5py==3.16.0
matplotlib==3.10.3
numpy==2.2.5
opencv-python==4.10.0
pandas==2.3.3
pyvips==3.1.1
safetensors==0.7.0
scanpy==1.12.2
scikit-learn==1.9.0
scipy==1.15.3
timm==1.0.27
torch==2.10.0
torchvision==0.25.0
transformers==5.12.1
```

The archived environment snapshot is available at [`../environment/cube_environment_snapshot.txt`](../environment/cube_environment_snapshot.txt). It records the complete shared environment used for provenance rather than serving as a minimal dependency specification.

## Data dependencies and order of execution

The expected split sizes are 3,717 training, 630 validation, and 945 test patches. Each split must use matching sample names across its pickle, UR1, and UR2 directories.

Run the scientific pipeline in this order:

1. generate split-specific pickle files with `data_preprocessing/preprocessing.py`;
2. train and evaluate `he_mihc/`;
3. export H&E-derived UR1 for train, validation, and test;
4. train and evaluate `he_st/` using the frozen training-set `st_stats.npz`;
5. export H&E-derived UR2 for train, validation, and test;
6. train and evaluate `ur/` and both single-UR ablations; and
7. run `analysis/` after all test prediction CSV files have been generated.

The required feature shapes are:

| Representation | Shape per patch |
|---|---:|
| UR1 | `[256, 32, 32]` |
| UR2 | `[256, 16, 16]` |
| fused summary representation | `[512]` |
| concept prediction | `[3]` |

Do not rename `model1`, `model2`, `model3`, or `model4`, or otherwise change model attribute names when loading the released checkpoints. These names are part of the checkpoint state-dictionary keys.

## Main commands

After configuring the corresponding `config.py` files:

```bash
# Final HE–mIHC branch
python -m he_mihc.trainer
python -m he_mihc.evaluate_pearson
python -m he_mihc.evaluate_recon
python -m he_mihc.visualize_recon
python -m he_mihc.export_ur1

# HE–pseudo-ST branch
python -m he_st.trainer
python -m he_st.evaluate_st
python -m he_st.diagnose_st
python -m he_st.visualize_st
python -m he_st.export_ur2

# UR fusion and ablations
python -m ur.trainer
python -m ur.evaluate
python -m ur1_ablation.trainer
python -m ur1_ablation.evaluate
python -m ur2_ablation.trainer
python -m ur2_ablation.evaluate

# Tissue QC and stratified evaluation
python -m analysis.tissue_qc
python -m analysis.tissue_stratified_evaluation
```

Detailed configuration, checkpoint selection, output files, and reported metrics are documented in the linked branch READMEs.
