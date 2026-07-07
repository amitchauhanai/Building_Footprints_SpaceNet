"""
config.py – single source of truth for all paths and hyperparameters.
Edit this file before running preprocess / train / evaluate / predict.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────
DATASET_ROOT = Path("/Users/amitchauhanai/SpaceNet/AOI_2_Vegas")

RGB_DIR       = DATASET_ROOT / "PS-RGB"          # PS-RGB GeoTIFFs
SOLUTIONS_CSV = DATASET_ROOT / "SN2_buildings_train_AOI_2_Vegas_solutions.csv"

# ─────────────────────────────────────────────
# Project output paths
# ─────────────────────────────────────────────
PROJECT_ROOT  = Path("/Users/amitchauhanai/SpaceNet/building_detection/unet_resnet34")

MASK_DIR      = PROJECT_ROOT / "masks"           # generated binary masks
SPLIT_DIR     = PROJECT_ROOT                     # train.txt / val.txt live here
CHECKPOINT_DIR= PROJECT_ROOT / "checkpoints"
LOG_DIR       = PROJECT_ROOT / "logs"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"         # visualisations / predictions

for _d in (MASK_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Image properties
# ─────────────────────────────────────────────
IMAGE_H = 650           # native PS-RGB height (px)
IMAGE_W = 650           # native PS-RGB width  (px)
IN_CHANNELS = 3         # RGB

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────
IMG_SIZE    = 512        # crop/resize fed to the network
BATCH_SIZE  = 8
NUM_EPOCHS  = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-5
LR_PATIENCE = 5         # ReduceLROnPlateau patience (epochs)
LR_FACTOR   = 0.5
EARLY_STOP  = 10        # stop if val IoU doesn't improve for N epochs

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
ENCODER         = "resnet34"
ENCODER_WEIGHTS = "imagenet"   # None to train from scratch
NUM_CLASSES     = 1            # binary segmentation

# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
BCE_WEIGHT  = 0.5       # weight of BCE in combined loss
DICE_WEIGHT = 0.5       # weight of Dice in combined loss
POS_WEIGHT  = 3.0       # BCE positive-class weight (buildings are rare)

# ─────────────────────────────────────────────
# Data split
# ─────────────────────────────────────────────
TRAIN_SPLIT = 0.85      # fraction of images used for training
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Inference / evaluation
# ─────────────────────────────────────────────
THRESHOLD   = 0.5       # sigmoid threshold for binary prediction
BEST_CKPT   = CHECKPOINT_DIR / "best_model.pth"

# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────
NUM_WORKERS = 4         # DataLoader workers
PIN_MEMORY  = True

# ImageNet normalisation (matches pre-trained ResNet34)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
