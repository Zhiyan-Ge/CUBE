# ============================================================
# HE-ST branch configuration
# ============================================================

# Data paths
TRAIN_DIR = "path/to/train/pkl"
VAL_DIR = "path/to/val/pkl"
ST_STATS_PATH = "path/to/st_stats.npz"
OUTPUT_DIR = "path/to/output"

# Sample counts: None = all; int = randomly select that many
TRAIN_SAMPLES = None
VAL_SAMPLES = None
USE_VALIDATION = True

# Model and device
GROUPS = 8
GPU_IDS = [3]

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 2026
USE_AMP = True

# Loss
ALPHA1 = 1.0     # HE: L1 + alpha1 * SSIM
BETA3 = 1.0      # Model3: HE -> ST
BETA4 = 0.1      # Model4: ST -> HE
GAMMA2 = 0.05    # UR2 weak alignment

# save
SAVE_EVERY = 10
SAVE_UR2 = True

TEST_DIR = "path/to/test/pkl"

# ============================================================
# HE-ST evaluation
# ============================================================

EVAL_CHECKPOINT = OUTPUT_DIR + "/best.pt"
EVAL_SPLIT = "test"          # "train" / "val" / "test"

EVAL_NUM_SAMPLES = 5        # visualize_st.py
EVAL_NUM_GENES = 8          # visualize_st.py
EVAL_SEED = SEED