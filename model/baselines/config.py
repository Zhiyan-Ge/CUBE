# ============================================================
# HEMIT 512 benchmarking configuration
# ============================================================

# Four supported benchmarks:
#   unet
#   resnet
#   pix2pix_unet
#   pix2pix_resnet
BENCHMARK = "pix2pix_resnet"

# CUBE preprocessed PKL directories
TRAIN_DIR = "path/to/train/pkl"
VAL_DIR = "path/to/val/pkl"
TEST_DIR = "path/to/test/pkl"

OUTPUT_ROOT = "path/to/output"
# Device / loader
GPU_ID = 3
BATCH_SIZE = 2
NUM_WORKERS = 8
SEED = 2026
USE_AMP = True

# Official HEMIT/pix2pix generator settings retained where applicable
NGF = 64
NDF = 64
NORM = "batch"
USE_DROPOUT = True
INIT_GAIN = 0.02

# Training protocol adapted from the HEMIT repository example
LR = 3e-5
BETA1 = 0.5
EPOCHS = 50
LR_STEP_EPOCH = 50
LR_GAMMA = 0.1

# Pix2pix settings from the supplied HEMIT repository
GAN_MODE = "lsgan"
LAMBDA_L1 = 30.0

# Logging / saving
LOG_EVERY = 50
SAVE_EVERY = 10

# Evaluation split 
EVAL_SPLIT = "test"  # "val" or "test"
