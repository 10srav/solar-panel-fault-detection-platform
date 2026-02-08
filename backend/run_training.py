"""
Quick launcher — run training from the project root.
Usage: python run_training.py
"""

import os
import sys

# Ensure solar_ai_system is the working directory for imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_rgb_model import build_model, train_model
from training.preprocessing import build_dataloaders
from config import DEVICE

if __name__ == "__main__":
    print("=" * 90)
    print(" SOLAR PANEL FAULT DETECTION — RGB MODEL TRAINING")
    print("=" * 90)
    print(f"Device: {DEVICE}\n")

    train_loader, val_loader, class_weights, class_names = build_dataloaders()
    model = build_model()
    history, best_acc = train_model(
        model, train_loader, val_loader, class_weights, class_names
    )
    print("\nDone. Start the API server next: python run_server.py")
