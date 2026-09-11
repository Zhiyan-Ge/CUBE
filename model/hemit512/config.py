# ============================================================
# HEMIT-512 adapted benchmark configuration
# ============================================================

# CUBE preprocessed PKL directories
TRAIN_DIR = "path/to/train/pkl"
VAL_DIR = "path/to/val/pkl"
TEST_DIR = "path/to/test/pkl"

OUTPUT_DIR = "path/to/output"
# Device / loader
GPU_ID = 3
BATCH_SIZE = 2
NUM_WORKERS = 8
SEED = 2026
USE_AMP = False  # IMPORTANT: official HEMIT training is FP32; fp16 AMP can poison this attention-heavy model

# HEMIT official generator settings.
# The only resolution adaptation is IMG_SIZE 1024 -> 512.
IMG_SIZE = 512
PATCH_SIZE = 32
WINDOW_SIZE = 64
EMBED_DIM = 96
DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (3, 6, 12, 24)
DROP_PATH_RATE = 0.2
NGF = 64
NDF = 64
N_RESBLOCKS = 6
TOP_K = 250  # 1024->512 halves each spatial axis; preserve original FMF top-k density (1000/4)
USE_DROPOUT = True
INIT_GAIN = 0.02

# Official README training command:
# lr=3e-5, lambda_L1=30, n_epochs=50, n_epochs_decay=30,
# lr_policy=step, batch_size=2, L1, no_flip.
LR = 3e-5
BETA1 = 0.5
EPOCHS = 80
LR_STEP_EPOCH = 50
LR_GAMMA = 0.1
GAN_MODE = "lsgan"
LAMBDA_L1 = 30.0

LOG_EVERY = 50
SAVE_EVERY = 10

# "val" or "test"
EVAL_SPLIT = "test"
