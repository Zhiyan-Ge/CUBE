# ============================================================
# HE-mIHC branch configuration
# ============================================================

# Data paths
TRAIN_DIR = "path/to/train/pkl"
VAL_DIR = "path/to/val/pkl"
OUTPUT_DIR = "path/to/output"
TEST_DIR = "path/to/test/pkl"

# Use validation set for evaluation during training
USE_VALIDATION = True

# Training and validation sample counts (None means use all available samples)
TRAIN_SAMPLES = None
VAL_SAMPLES = None

# Model configuration
GROUPS = 8
GPU_IDS = [2, 3]

# Training configuration
EPOCHS = 100
# Batch size for each GPU (global batch = BATCH_SIZE * len(GPU_IDS) in DDP mode)
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 2026
USE_AMP = True

# Reconstruction loss
ALPHA1 = 1.0     # L_HE = L1 + alpha1 * SSIM
ALPHA2 = 1.0     # L_mIHC(channel) = L1 + alpha2 * SSIM

# HE -> mIHC Pearson correlation supervision
MIHC_PEARSON_WEIGHT = 0.75

# Multi-scale Pearson configuration (scales: 1=512,2=256,4=128)
MIHC_PEARSON_SCALES = [1, 2, 4]
MIHC_PEARSON_SCALE_WEIGHTS = [0.5, 0.3, 0.2]

# CD3 soft Dice foreground weight (only for HE->mIHC channel 1)
CD3_DICE_WEIGHT = 0.2

# Model / branch loss
BETA1 = 1.0      # HE -> mIHC
BETA2 = 1.0      # mIHC -> HE
GAMMA1 = 0.05    # UR1 weak alignment

# LOG & SAVE configuration
LOG_EVERY = 10
SAVE_EVERY = 10

SAVE_UR1 = True

# mIHC foreground-aware reconstruction
MIHC_FG_THRESHOLD = 0.05
MIHC_FG_EXTRA_WEIGHT = 2.0

# Channel weights for mIHC reconstruction loss (for channels 1, 2, 3)
MIHC_CHANNEL_WEIGHTS = [1.0, 2.0, 1.0]

# paired spatial augmentation
USE_PAIRED_AUG = True