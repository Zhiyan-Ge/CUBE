# ============================================================
# UR branch configuration
# ============================================================

# Training data
TRAIN_PKL_DIR = "path/to/train/pkl"
TRAIN_UR1_DIR = "path/to/train/ur1"
TRAIN_UR2_DIR = "path/to/train/ur2"

# Validation data
VAL_PKL_DIR = "path/to/val/pkl"
VAL_UR1_DIR = "path/to/val/ur1"
VAL_UR2_DIR = "path/to/val/ur2"

# Test data
TEST_PKL_DIR = "path/to/test/pkl"
TEST_UR1_DIR = "path/to/test/ur1"
TEST_UR2_DIR = "path/to/test/ur2"

# Output directory
OUTPUT_DIR = "path/to/output"

# Data loading, if None, use all samples
TRAIN_SAMPLES = None
VAL_SAMPLES = None
USE_VALIDATION = True

# Model configuration
DIM = 256
HEADS = 8
FFN_DIM = 512
DROPOUT = 0.1

# Training configuration
GPU_IDS = [3]
EPOCHS = 80
BATCH_SIZE = 4
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 2026
USE_AMP = True

# Checkpoint saving
SAVE_EVERY = 10

# Evaluation
EVAL_CHECKPOINT = OUTPUT_DIR + "/best.pt"
EVAL_SPLIT = "test"   # "val" / "test"