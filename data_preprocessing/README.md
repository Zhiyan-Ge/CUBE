# CUBE data preprocessing

This directory converts paired H&E and multiplex immunohistochemistry (mIHC) patches into the serialized inputs used by CUBE. It also generates a DeepSpot-derived spatial pseudo-transcriptomic matrix for the HE–ST branch.

The preprocessing code is dataset-specific. The released pipeline was used with 1,024 × 1,024 HEMIT patches and preserves the numerical behavior of the experiments reported with CUBE.

## Pipeline

For each paired H&E/mIHC patch, `preprocessing.py` performs the following operations:

1. reads the paired 1,024 × 1,024 TIFF images;
2. parses the global and patch-grid coordinates from the filename;
3. computes three mIHC concept scores from the original 1,024 × 1,024 channels;
4. uses DeepSpot to predict a 16 × 16 spatial pseudo-ST matrix containing 256 genes;
5. downsamples H&E to 512 × 512 and 256 × 256, and mIHC to 512 × 512;
6. scales image intensities to `[0, 1]`; and
7. writes one pickle file per paired patch and a dataset-level preprocessing report.

The mIHC channel order used throughout this repository is:

| Channel index | Marker |
|---:|---|
| 0 | DAPI |
| 1 | CD3 |
| 2 | panCK |

Each concept score is the fraction of pixels above the channel-specific Otsu threshold in the full-resolution mIHC patch. These are continuous values in `[0, 1]`; they are not cell counts.

## Input organization

H&E and mIHC images must be stored in separate directories with identical TIFF filenames:

```text
HE/
├── [6407,49798]_patch_0_0.tif
└── [6407,49798]_patch_0_1.tif

mIHC/
├── [6407,49798]_patch_0_0.tif
└── [6407,49798]_patch_0_1.tif
```

Filenames are expected to contain both a global coordinate in square brackets and a `_patch_i_j` grid suffix. The current code normalizes the global coordinates by the HEMIT-specific constants `WSI_WIDTH_BASE = 22541` and `WSI_HEIGHT_BASE = 64216`. Change these constants only when adapting the pipeline to a dataset with different coordinate conventions; doing so changes the coordinate input used by the UR model.

The script lists files from the H&E directory. A corresponding mIHC image must exist for every selected filename.

## DeepSpot requirements

Pseudo-ST generation depends on an external DeepSpot checkout and its pretrained resources.

The released CUBE preprocessing used the following resources:

- DeepSpot repository: [ratschlab/DeepSpot](https://github.com/ratschlab/DeepSpot);
- DeepSpot pretrained model archive: [Zenodo record 15322099](https://zenodo.org/records/15322099);
- pretrained expression model: `DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl`;
- matching gene-information file: `DeepSpot_pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv`;
- matching hyperparameter file: `DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml`;
- UNI morphology-model weights from [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI).

The complete DeepSpot pretrained-model archive can be downloaded from:

<https://zenodo.org/records/15322099/files/DeepSpot_pretrained_model_weights.zip?download=1>

The archived hyperparameter file used for the CUBE preprocessing run contains:

```yaml
augmentation: aestetik
batch_size: 1024
epochs: 15
gene_norm: standard
image_feature_model: uni
neighbors: 1
res: 1
spot_context: spot_subspot_neighbors
```

## Configuration

The main paths can be supplied as environment variables:

| Environment variable | Purpose |
|---|---|
| `CUBE_HE_DIR` | directory containing H&E TIFF patches |
| `CUBE_MIHC_DIR` | directory containing paired mIHC TIFF patches |
| `CUBE_PREPROCESSING_OUTPUT` | output directory for pickle files and the report |
| `DEEPSPOT_REPO_PATH` | DeepSpot repository path |
| `DEEPSPOT_MODEL_WEIGHTS_PATH` | DeepSpot expression-model weights |
| `DEEPSPOT_MODEL_HPARAM_PATH` | DeepSpot hyperparameter YAML |
| `DEEPSPOT_GENE_INFO_CSV_PATH` | DeepSpot gene-information CSV |
| `DEEPSPOT_MORPHOLOGY_MODEL_PATH` | UNI morphology-model weights |

GPU selection and the fixed preprocessing constants remain in `preprocessing.py`. The released settings are `GRID_SIZE = 16`, `TARGET_GENE_COUNT = 256`, `N_MINI_TILES = 9`, `NEIGHBOR_RADIUS = 1`, `FILTER_WHITE = False`, and `CLIP_NEGATIVE = False`.

## Run

Run the entry point from this directory because it imports the numbered helper modules as local modules:

```bash
cd data_preprocessing

export CUBE_HE_DIR=/path/to/HE
export CUBE_MIHC_DIR=/path/to/mIHC
export CUBE_PREPROCESSING_OUTPUT=/path/to/preprocessed/all
export DEEPSPOT_REPO_PATH=/path/to/DeepSpot
export DEEPSPOT_MODEL_WEIGHTS_PATH=/path/to/final_model.pkl
export DEEPSPOT_MODEL_HPARAM_PATH=/path/to/top_param_overall.yaml
export DEEPSPOT_GENE_INFO_CSV_PATH=/path/to/info_highly_variable_genes.csv
export DEEPSPOT_MORPHOLOGY_MODEL_PATH=/path/to/pytorch_model.bin

python preprocessing.py
```

The released study directly followed the original HEMIT dataset split, containing 3,717 training, 630 validation, and 945 test patches. No additional random splitting or reassignment was performed by CUBE. The preprocessing script itself does not assign dataset splits; preprocess the samples while preserving the original HEMIT train/validation/test membership.

## Pickle schema

Each output pickle contains the following dictionary:

| Key | Type and shape | Description |
|---|---|---|
| `he_matrix_512` | `float16 [3, 512, 512]` | H&E image scaled to `[0, 1]` |
| `mihc_matrix_512` | `float16 [3, 512, 512]` | mIHC image scaled to `[0, 1]` |
| `he_matrix_256` | `float16 [3, 256, 256]` | H&E input for the HE–ST branch |
| `st_matrix` | `float32 [16, 16, 256]` | DeepSpot-derived spatial pseudo-ST matrix |
| `concept_scores` | `float32 [3]` | DAPI, CD3, and panCK Otsu-positive fractions |
| `sample_name` | `str` | filename without extension |
| `normalized_mega_coord` | `float32 [2]` | normalized global patch coordinate |
| `patch_grid_coord` | `int64 [2]` | parsed `_patch_i_j` coordinate |
| `metadata` | `dict` | reserved metadata dictionary |

`preprocessing_report.json` records processed/failed sample counts, shapes, dtypes, summary statistics, timing, selected gene names, and the gene-wise pseudo-ST mean and standard deviation.


## HE–ST normalization statistics

After organizing the generated pickle files into the frozen train/validation/test splits, generate the pseudo-ST normalization statistics from the training split only:

```bash
python generate_st_stats.py \
    /path/to/train/pkl \
    --output /path/to/train/st_stats.npz
```
generate_st_stats.py reads st_matrix from every training pickle, reshapes each matrix from [16, 16, 256] to [256, 256], and accumulates gene-wise sums and squared sums in float64. It then saves the population mean and standard deviation for each of the 256 genes. The script does not rerun DeepSpot and does not modify or clip the stored pseudo-ST matrices.

The generated `st_stats.npz` contains:

| Key | Shape | Meaning |
|---|---:|---|
| `mean` | `[256]` | Training-set gene-wise mean |
| `raw_std` | `[256]` | Training-set gene-wise population standard deviation |
| `safe_std` | `[256]` | Standard deviation used for Z-score normalization; values below `eps` are replaced with `1.0` |
| `total_spots` | scalar | Total number of accumulated 16 × 16 spatial positions |
| `num_files` | scalar | Number of training pickle files |
| `eps` | scalar | Threshold used to identify near-zero standard deviations |

For the released CUBE split, the script should report:
Files: 3717
Total spots: 951552

The archived statistics were computed from 3,717 training files, corresponding to 3717 × 16 × 16 = 951552 spatial positions. Do not use a newly generated statistics file if these counts do not match the intended training split.

Never compute pseudo-ST normalization statistics from validation or test data. Validation and test samples must use the training-set mean and safe_std. The statistics file and all pseudo-ST matrices must also use exactly the same 256-gene order.

The supplied archived st_stats.npz should be used for exact reproduction of the released experiments. generate_st_stats.py is provided to regenerate the same statistics when the original training pickle files are available. The gene-wise mean and standard deviation recorded in preprocessing_report.json can be used for additional numerical verification.

## Interpretation

The generated `st_matrix` is a DeepSpot-derived pseudo-transcriptomic target predicted from H&E. It is not independently measured spatial transcriptomics. Results from the HE–ST branch therefore assess reconstruction of this frozen pseudo-ST representation and must not be described as validation against an independent transcriptomic assay.

Continue with the [model documentation](../model/README.md) after generating the split-specific pickle directories and the training-set `st_stats.npz`.
