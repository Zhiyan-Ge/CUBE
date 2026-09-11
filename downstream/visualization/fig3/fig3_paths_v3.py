from pathlib import Path

# -----------------------------------------------------------------------------
# CUBE HE-ST paths supplied for the frozen Fig3 experiment
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "model"
HE_ST_CODE_DIR = MODEL_ROOT / "he_st"
HE_ST_CHECKPOINT = PROJECT_ROOT / "models/test_models_3/test13/he_st/best.pt"
HE_ST_RUN_DIR = HE_ST_CHECKPOINT.parent
TEST_PKL_DIR = PROJECT_ROOT / "data/train_data/final_data/test/pkl"
ST_STATS_PATH = PROJECT_ROOT / "data/train_data/final_data/train/st_stats.npz"
HE_ST_EVAL_JSON = HE_ST_RUN_DIR / "evaluation/best_test/st_evaluation.json"

# -----------------------------------------------------------------------------
# DeepSpot paths
# -----------------------------------------------------------------------------
DEEPSPOT_MODEL_DIR = PROJECT_ROOT / "data_preprocessing/DeepSpot_model"
DEEPSPOT_REPO_DIR = PROJECT_ROOT / "data_preprocessing/DeepSpot/DeepSpot-main"
DEEPSPOT_WRAPPER = PROJECT_ROOT / "data_preprocessing/_04_ST_processing.py"
DEEPSPOT_MODEL_WEIGHTS = DEEPSPOT_MODEL_DIR / "final_model.pkl"
DEEPSPOT_HPARAMS = DEEPSPOT_MODEL_DIR / "top_param_overall.yaml"
DEEPSPOT_HVG = DEEPSPOT_MODEL_DIR / "info_highly_variable_genes.csv"
DEEPSPOT_MORPHOLOGY_WEIGHTS = DEEPSPOT_MODEL_DIR / "pytorch_model.bin"

# -----------------------------------------------------------------------------
# Final Visium HD experiment paths recovered from the frozen real-ST scripts
# -----------------------------------------------------------------------------
REAL_ST_ROOT = PROJECT_ROOT / "result" / "01_real_st"
HD_DATA_ROOT = REAL_ST_ROOT / "data/HD"
PAIRED_HD_DIR = HD_DATA_ROOT / "visium_hd_16um_paired"
MACENKO_HD_DIR = HD_DATA_ROOT / "visium_hd_16um_macenko"
FINETUNE_PREP_DIR = HD_DATA_ROOT / "visium_hd_cube_finetune/prepared"
DECODER_FT_DIR = HD_DATA_ROOT / "visium_hd_cube_finetune/decoder_only_100"
DEEPSPOT_TEST_DIR = HD_DATA_ROOT / "visium_hd_deepspot_spatial_test"

PATCH_METADATA = PAIRED_HD_DIR / "patch_metadata.csv"
GENE_MAPPING = PAIRED_HD_DIR / "gene_mapping.csv"
ST_TARGET_DIR = PAIRED_HD_DIR / "st_targets"
NORMALIZED_HD_PATCH_DIR = MACENKO_HD_DIR / "patches_1024"
SPATIAL_SPLIT_CSV = FINETUNE_PREP_DIR / "spatial_split.csv"
UR2_FEATURE_DIR = FINETUNE_PREP_DIR / "ur2_features"
FINETUNING_PATCH_SPLIT = DECODER_FT_DIR / "finetuning_patch_split.csv"
REAL_ST_TRAIN_STATS = DECODER_FT_DIR / "real_st_train_stats.npz"

PRETRAINED_DECODER = DECODER_FT_DIR / "best_decoder_pretrained.pt"
SCRATCH_DECODER = DECODER_FT_DIR / "best_decoder_scratch.pt"
RESET_HEAD_DECODER = DECODER_FT_DIR / "best_decoder_reset_head.pt"

FINETUNING_SUMMARY = DECODER_FT_DIR / "finetuning_summary.csv"
RESET_HEAD_SUMMARY = DECODER_FT_DIR / "reset_head_summary.csv"
DEEPSPOT_SUMMARY = DEEPSPOT_TEST_DIR / "deepspot_spatial_test_summary.csv"

GENE_METRICS = {
    "deepspot": DEEPSPOT_TEST_DIR / "gene_metrics_normalized.csv",
    "zero_shot": DECODER_FT_DIR / "gene_metrics_zero_shot.csv",
    "pretrained": DECODER_FT_DIR / "gene_metrics_pretrained.csv",
    "scratch": DECODER_FT_DIR / "gene_metrics_scratch.csv",
    "reset_head": DECODER_FT_DIR / "gene_metrics_reset_head.csv",
}

# -----------------------------------------------------------------------------
# Fig3 visualization paths
# -----------------------------------------------------------------------------
FIG3_CODE_DIR = Path(__file__).resolve().parent
FIG3_RESULT_DIR = PROJECT_ROOT / "result" / "visualization" / "fig3" / "result"
FIG3_SOURCE_DIR = FIG3_RESULT_DIR / "source_data"
FIG3_CACHE_DIR = FIG3_RESULT_DIR / "cache"

# Creating output directories is normal initialization, not a data/path check.
FIG3_RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG3_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
FIG3_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Candidate genes for the first Fig3F draft. Final choices can be changed after
# inspecting the whole-slide maps.
FIG3F_GENES = ["SPARC", "CEACAM6", "VIM"]

GPU_ID = 3
GROUPS = 8
SEED = 2026
