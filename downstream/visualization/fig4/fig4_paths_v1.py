from pathlib import Path


# =============================================================================
# Repository root and input paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TISSUE_CSV = (
    PROJECT_ROOT / "models" / "test_models_3" / "test15" / "ur_branch" /
    "tissue_qc" / "tissue_fraction_per_patch.csv"
)

FUSION_TEST_PRED_CSV = (
    PROJECT_ROOT / "models" / "test_models_3" / "test15" / "ur_branch" /
    "evaluation" / "best_test" / "predictions.csv"
)
UR1_TEST_PRED_CSV = (
    PROJECT_ROOT / "models" / "ur1_ablation" /
    "evaluation" / "best_test" / "predictions.csv"
)
UR2_TEST_PRED_CSV = (
    PROJECT_ROOT / "models" / "ur2_ablation" /
    "evaluation" / "best_test" / "predictions.csv"
)

FUSION_VAL_PRED_CSV = (
    PROJECT_ROOT / "models" / "test_models_3" / "test15" / "ur_branch" /
    "evaluation" / "best_val" / "predictions.csv"
)
UR1_VAL_PRED_CSV = (
    PROJECT_ROOT / "models" / "ur1_ablation" /
    "evaluation" / "best_val" / "predictions.csv"
)
UR2_VAL_PRED_CSV = (
    PROJECT_ROOT / "models" / "ur2_ablation" /
    "evaluation" / "best_val" / "predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "result" / "visualization" / "fig4" / "result" / "tissue_stratified"


# =============================================================================
# Shared settings
# =============================================================================

CONCEPTS = ["DAPI", "CD3", "panCK"]
MODELS = ["UR1-only", "UR2-only", "Fusion"]
THRESHOLDS = [0.00, 0.01, 0.05, 0.10, 0.25, 0.30, 0.50, 0.75]
THRESHOLD_LABELS = ["Full", "≥1%", "≥5%", "≥10%", "≥25%", "≥30%", "≥50%", "≥75%"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
