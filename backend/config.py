"""
=============================================================================
 SOLAR AI SYSTEM — CENTRAL CONFIGURATION
 All paths, hyperparameters, and constants in one place.
=============================================================================
"""

import os
import torch

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "PRoject")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rgb_fault_model.pth")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STATIC_DIR = os.path.join(BASE_DIR, "api", "static")

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA PIPELINE
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
SPLIT_RATIO = 0.8
RANDOM_SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
CLASS_NAMES = [
    "Bird_drop_generateds",
    "Clean",
    "Dusty",
    "Electrical_damage_generated",
    "Physcial_damage_generated",
    "Snow_covered_generated",
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE = 1e-4
EPOCHS = 15
STEP_SIZE = 5
GAMMA = 0.5
EARLY_STOPPING_PATIENCE = 5

# Layers to freeze in ResNet18 (early feature extractors)
FREEZE_LAYERS = ["conv1", "bn1", "layer1", "layer2"]

# ============================================================================
# GRAD-CAM
# ============================================================================
GRADCAM_THRESHOLD = 0.5
GRADCAM_OVERLAY_ALPHA = 0.4

# ============================================================================
# RISK ENGINE
# ============================================================================
RISK_THRESHOLDS = {
    "low_max": 30,
    "medium_max": 60,
}

MAINTENANCE_SUGGESTIONS = {
    "Bird_drop_generateds":        "Cleaning required — remove bird droppings from panel surface",
    "Clean":                       "No action needed — panel is in good condition",
    "Dusty":                       "Surface cleaning — remove dust accumulation for optimal output",
    "Electrical_damage_generated": "Electrical inspection — check wiring/connections immediately",
    "Physcial_damage_generated":   "Panel replacement — physical damage detected, replace panel",
    "Snow_covered_generated":      "Snow removal — clear snow/ice from panel surface",
}

# ============================================================================
# API
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 20

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
