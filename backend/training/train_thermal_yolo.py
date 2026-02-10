"""
=============================================================================
 THERMAL FAULT DETECTION — YOLOv8 TRAINING
 Object detection for thermal imaging fault localization.
=============================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# YOLOv8 will be installed if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("[INSTALL] Installing ultralytics (YOLOv8)...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

from config import DEVICE

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
THERMAL_DATASET = PROJECT_ROOT / "dataset" / "Thermal Imaging.v6i.yolov8"
DATA_YAML = THERMAL_DATASET / "data.yaml"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_SAVE_PATH = MODEL_DIR / "thermal_yolo.pt"

# Training hyperparameters
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 0.01

# ============================================================================
# VERIFY DATASET
# ============================================================================

def verify_dataset():
    """Check if thermal dataset exists and is properly formatted."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}\n"
            f"Ensure thermal dataset is at: {THERMAL_DATASET}"
        )

    # Check train/val/test folders
    for split in ['train', 'valid', 'test']:
        img_dir = THERMAL_DATASET / split / 'images'
        lbl_dir = THERMAL_DATASET / split / 'labels'

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing: {lbl_dir}")

        num_images = len(list(img_dir.glob('*.jpg')))
        num_labels = len(list(lbl_dir.glob('*.txt')))

        print(f"[{split.upper():5}] {num_images} images, {num_labels} labels")

    print(f"\n✅ Dataset verified: {THERMAL_DATASET}")
    print(f"   data.yaml: {DATA_YAML}")

# ============================================================================
# TRAIN YOLOV8
# ============================================================================

def train_thermal_yolo():
    """
    Train YOLOv8n (nano) on thermal imaging dataset.

    Classes:
        0: Connectivity Issue/Loose Connection/Corroded
        1: Deteriorated Insulation
        2: Faulty Circuit Breakers
        3: Normal Operation Condition
        4: Overloaded Circuits
        5: Phase Imbalance
    """
    print("\n" + "="*80)
    print(" THERMAL FAULT DETECTION — YOLOv8 TRAINING")
    print("="*80)

    # Verify dataset
    verify_dataset()

    # Load pretrained YOLOv8n model
    print(f"\n[MODEL] Loading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')

    # Check device
    device = 0 if DEVICE.type == 'cuda' else 'cpu'
    print(f"[DEVICE] Using: {DEVICE}")

    # Training configuration
    print(f"\n[TRAINING] Configuration:")
    print(f"   Epochs:      {EPOCHS}")
    print(f"   Image size:  {IMG_SIZE}")
    print(f"   Batch size:  {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Device:      {device}")

    # Start training
    print(f"\n{'='*80}")
    print(" TRAINING START")
    print(f"{'='*80}\n")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        device=device,
        project=str(MODEL_DIR.parent / 'training' / 'thermal_runs'),
        name='yolov8_thermal',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        save=True,
        plots=True,
    )

    print(f"\n{'='*80}")
    print(" TRAINING COMPLETE")
    print(f"{'='*80}")

    # Get best model path
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'

    # Copy to models directory
    import shutil
    shutil.copy(best_model_path, MODEL_SAVE_PATH)

    print(f"\n✅ Best model saved to: {MODEL_SAVE_PATH}")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    return results

# ============================================================================
# VALIDATE MODEL
# ============================================================================

def validate_thermal_yolo(model_path=None):
    """Run validation on the trained model."""
    if model_path is None:
        model_path = MODEL_SAVE_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n[VALIDATION] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Validate
    metrics = model.val(data=str(DATA_YAML))

    print(f"\n📊 Validation Metrics:")
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")

    return metrics

# ============================================================================
# TEST INFERENCE
# ============================================================================

def test_inference(image_path=None, model_path=None):
    """Test inference on a sample thermal image."""
    if model_path is None:
        model_path = MODEL_SAVE_PATH

    if image_path is None:
        # Use first image from test set
        test_img_dir = THERMAL_DATASET / 'test' / 'images'
        images = list(test_img_dir.glob('*.jpg'))
        if not images:
            print("No test images found")
            return
        image_path = images[0]

    print(f"\n[TEST] Running inference on: {image_path.name}")

    model = YOLO(str(model_path))
    results = model(str(image_path))

    # Display results
    for result in results:
        boxes = result.boxes
        print(f"\n   Detected {len(boxes)} fault(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f"     - {class_name}: {conf:.1%}")

    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 on thermal imaging dataset')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    parser.add_argument('--test-only', action='store_true', help='Only run test inference')
    args = parser.parse_args()

    if args.validate_only:
        validate_thermal_yolo()
    elif args.test_only:
        test_inference()
    else:
        # Full training
        results = train_thermal_yolo()

        # Validate
        print("\n" + "="*80)
        validate_thermal_yolo()

        # Test inference
        test_inference()

        print("\n✅ Thermal YOLOv8 training complete!")
        print(f"   Model: {MODEL_SAVE_PATH}")
        print("\n   Next step: python train_thermal_unet.py")
