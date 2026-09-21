"""
config.py
---------

"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# DATA_DIR must point at the folder that directly contains the five class
# sub-folders (Healthy, Mild DR, Moderate DR, Proliferate DR, Severe DR).
DATA_DIR = os.environ.get("DR_DATA_DIR", "/content/data")

OUTPUT_DIR = os.environ.get("DR_OUTPUT_DIR", "/content/outputs")
MODEL_DIR = os.environ.get("DR_MODEL_DIR", "/content/saved_models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in (OUTPUT_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Exact folder names as they appear in the Kaggle "Diabetic Retinopathy"
# balanced dataset. 
CLASS_NAMES = [
    "Healthy",
    "Mild DR",
    "Moderate DR",
    "Proliferate DR",
    "Severe DR",
]
NUM_CLASSES = len(CLASS_NAMES)

# Original images in the dataset are 256x256 RGB PNGs.
RAW_IMAGE_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)          # standard input size for most ImageNet backbones
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15               # taken out of the remaining (train+val) portion
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS_HEAD = 15               # phase 1: train the classification head only
EPOCHS_FINE_TUNE = 20          # phase 2: unfreeze top backbone layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
FINE_TUNE_AT_LAYER = -40       # unfreeze the last 40 layers of the backbone

BACKBONE = "EfficientNetB0"    # see model.py -> BACKBONE_REGISTRY for options

EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
