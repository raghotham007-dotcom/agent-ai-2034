"""
Quick-Fix Configuration for ApexLoop Pipeline
Apply these changes to achieve immediate improvements.

This file contains the recommended settings for Phase 1 fixes.
"""

# ====================
# IMAGE RESOLUTION
# ====================
# BEFORE: 64x64 (too small for defect detection)
# AFTER: 224x224 (standard for ResNet, preserves detail)
IMAGE_SIZE = 224

# ====================
# MODEL CONFIGURATION
# ====================
# Option 1: Use smaller model (RECOMMENDED)
USE_MOBILENET = True  # If False, uses ResNet18 with freezing

# Option 2: ResNet18 with layer freezing
FREEZE_LAYERS = True
FREEZE_PERCENTAGE = 0.75  # Freeze first 75% of parameters

# Dropout for regularization
DROPOUT_RATE = 0.7  # Increased from 0.5

# Weight decay
WEIGHT_DECAY = 1e-3  # Increased from 1e-4

# ====================
# DATA AUGMENTATION
# ====================
# Disable file-based augmentation (rely on runtime augmentation)
FILE_AUGMENTATION_COPIES = 0  # Was 2

# Runtime augmentation settings
USE_RUNTIME_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
}

# ====================
# CLASS WEIGHTS
# ====================
# Stronger class weight damping
CLASS_WEIGHT_DAMPING = 0.8  # Was 0.5 (sqrt). 1.0 = linear, 0.5 = sqrt

# ====================
# DATA CLEANING
# ====================
# More realistic Cleanlab thresholds
CLEANLAB_THRESHOLDS = {
    'standard': {
        'prob_threshold_correct': 0.05,  # Was 0.01
        'prob_threshold_wrong': 0.90,    # Was 0.99
    },
    'aggressive': {
        'prob_threshold_correct': 0.15,  # Was 0.2
        'prob_threshold_wrong': 0.75,    # Was 0.7
    }
}

# ====================
# TRAINING SETTINGS
# ====================
# Keep existing settings for now
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2
LR_SCHEDULER_PATIENCE = 1

# ====================
# SUMMARY
# ====================
print("=" * 60)
print("ApexLoop Quick-Fix Configuration Loaded")
print("=" * 60)
print(f"Image Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Model: {'MobileNetV2' if USE_MOBILENET else 'ResNet18'}")
if not USE_MOBILENET:
    print(f"  - Freeze Layers: {FREEZE_LAYERS} ({FREEZE_PERCENTAGE*100:.0f}%)")
print(f"  - Dropout: {DROPOUT_RATE}")
print(f"  - Weight Decay: {WEIGHT_DECAY}")
print(f"Runtime Augmentation: {USE_RUNTIME_AUGMENTATION}")
print(f"File Augmentation Copies: {FILE_AUGMENTATION_COPIES}")
print(f"Class Weight Damping: {CLASS_WEIGHT_DAMPING}")
print(f"Cleanlab Threshold (standard): {CLEANLAB_THRESHOLDS['standard']}")
print("=" * 60)
