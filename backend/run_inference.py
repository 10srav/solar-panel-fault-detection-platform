"""
Quick launcher — run inference on images from CLI.
Usage: python run_inference.py <image_path> [image2 ...]
       python run_inference.py              (runs demo)
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.predictor import load_model, analyze_image, print_result
from config import CLASS_NAMES, DATASET_PATH, VALID_EXTENSIONS

if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) < 2:
        print("\nUsage: python run_inference.py <image_path> [image2 ...]")
        print("\nRunning demo — one sample per class:\n")

        for cls in CLASS_NAMES:
            cls_dir = os.path.join(DATASET_PATH, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = [f for f in os.listdir(cls_dir)
                    if f.lower().endswith(tuple(VALID_EXTENSIONS))]
            if imgs:
                path = os.path.join(cls_dir, imgs[0])
                print(f"\n{'='*60}")
                print(f" ANALYZING: {os.path.basename(path)}")
                print(f"{'='*60}")
                result = analyze_image(path, model=model)
                print_result(result)
    else:
        for path in sys.argv[1:]:
            print(f"\n{'='*60}")
            print(f" ANALYZING: {os.path.basename(path)}")
            print(f"{'='*60}")
            try:
                result = analyze_image(path, model=model)
                print_result(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
