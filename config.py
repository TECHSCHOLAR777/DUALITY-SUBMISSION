"""
config.py  —  Duality AI Offroad Segmentation
All hyperparameters and paths in one place.
"""

import os

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME = "duality-offroad-segmentation"
BASE_PATH    = f"data"          # local: data/train, data/val, data/test
TRAIN_DIR    = os.path.join(BASE_PATH, "train")
VAL_DIR      = os.path.join(BASE_PATH, "val")
TEST_DIR     = os.path.join(BASE_PATH, "test")

# Kaggle path override (used automatically when running on Kaggle)
KAGGLE_BASE  = f"/kaggle/input/{DATASET_NAME}/data"

# ── Class mapping ─────────────────────────────────────────────────────────────
VALUE_MAP = {
    0: 0,      # Background
    100: 1,    # Trees
    200: 2,    # Lush Bushes
    300: 3,    # Dry Grass
    500: 4,    # Dry Bushes
    550: 5,    # Ground Clutter
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9   # Sky
}
NUM_CLASSES  = len(VALUE_MAP)
CLASS_NAMES  = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

# Colour palette for mask visualisation (RGB)
PALETTE = [
    [0,   0,   0  ], [34,  139, 34 ], [0,   200, 0  ],
    [210, 180, 140], [139, 90,  43 ], [128, 128, 0  ],
    [139, 69,  19 ], [128, 128, 128], [160, 82,  45 ],
    [135, 206, 235]
]

# ── Model ─────────────────────────────────────────────────────────────────────
ENCODER_NAME    = "resnet50"
ENCODER_WEIGHTS = "imagenet"

# ── Training ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 768
BATCH_SIZE   = 4
EPOCHS       = 30
LR           = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE     = 6        # early stopping patience
CKPT_FREQ    = 5        # save checkpoint every N epochs
NUM_WORKERS  = 2

# Class weights (emphasise rare classes)
CLASS_WEIGHTS = [1, 2, 4, 2, 3, 5, 6, 6, 1, 1]

# ── Tversky Loss ──────────────────────────────────────────────────────────────
TVERSKY_ALPHA = 0.7
TVERSKY_BETA  = 0.3

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = "outputs"
CKPT_DIR       = "checkpoints"
BEST_MODEL     = "best_model.pth"
FINAL_MODEL    = "final_model.pth"
HISTORY_JSON   = os.path.join(OUTPUT_DIR, "training_history.json")
