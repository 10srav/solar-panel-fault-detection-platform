"""
=============================================================================
 THERMAL SEGMENTATION — U-Net TRAINING
 Pixel-level fault segmentation for thermal images.
=============================================================================
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import time

# ============================================================================
# GPU ENFORCEMENT — NO CPU FALLBACK
# ============================================================================

assert torch.cuda.is_available(), "CUDA GPU REQUIRED -- Training stopped"

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # cuDNN autotuner for fixed input sizes

gpu_name = torch.cuda.get_device_name(0)
assert "NVIDIA" in gpu_name or "RTX" in gpu_name, f"Expected NVIDIA GPU, got: {gpu_name}"
print(f"Using GPU: {gpu_name}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN benchmark: ON")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
THERMAL_DATASET = PROJECT_ROOT / "dataset" / "Thermal Imaging.v6i.yolov8"
MASKS_DIR = THERMAL_DATASET / "masks"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_SAVE_PATH = MODEL_DIR / "thermal_segmentation_unet_v2.pth"

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 256
BATCH_SIZE = 16  # Push GPU memory to ~3.8GB
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
NUM_WORKERS = 6

# ============================================================================
# U-NET ARCHITECTURE
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block: (Conv → BN → ReLU) × 2 with Dropout"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Initialize weights with Kaiming (He) initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary segmentation.
    Input: [B, 3, 256, 256] RGB image
    Output: [B, 1, 256, 256] binary mask
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output (no sigmoid - will use BCEWithLogitsLoss)
        return self.out(dec1)

# ============================================================================
# DATASET
# ============================================================================

class ThermalDataset(Dataset):
    """Thermal imaging dataset with images and masks + augmentation."""

    def __init__(self, split='train', img_size=256, augment=True):
        self.img_size = img_size
        self.split = split
        self.augment = augment and (split == 'train')  # Only augment training data

        # Paths
        self.img_dir = THERMAL_DATASET / split / 'images'
        self.mask_dir = MASKS_DIR / split

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Masks not found at {self.mask_dir}\n"
                f"Run generate_thermal_masks.py first!"
            )

        # Get all images
        self.images = sorted(list(self.img_dir.glob('*.jpg')))

        # Filter: only keep images that have masks
        self.images = [
            img for img in self.images
            if (self.mask_dir / (img.stem + '.png')).exists()
        ]

        print(f"[{split.upper()}] Loaded {len(self.images)} images with masks")

    def __len__(self):
        return len(self.images)

    def apply_augmentation(self, image, mask):
        """Apply random augmentations to image and mask."""
        import random

        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)

        # Contrast adjustment
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

        return image, mask

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        # Apply augmentation
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)  # [H, W] → [1, H, W]

        return image, mask

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Improved Dice loss for segmentation (works with logits)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(logits)

        # Flatten predictions and targets
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss with logits (safe for autocast)."""

    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # Use BCEWithLogitsLoss (safe for autocast)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        pt = torch.where(target == 1, prob, 1 - prob)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Focal + Dice loss for robust segmentation (autocast-safe)."""

    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.8, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)

    def forward(self, logits, target):
        return self.focal(logits, target) + self.dice(logits, target)

# ============================================================================
# TRAINING
# ============================================================================

def train_unet():
    """Train U-Net on thermal masks."""

    print("="*80)
    print(" THERMAL SEGMENTATION — U-Net TRAINING")
    print("="*80)

    # Verify GPU training
    print(f"\n[GPU] TRAINING ON: {torch.cuda.get_device_name(0)} -- NVIDIA RTX 3050")
    print(f"      Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"      Memory Reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Build datasets
    train_dataset = ThermalDataset('train', IMG_SIZE)
    val_dataset = ThermalDataset('valid', IMG_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    print(f"\n[DATA] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Build model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] U-Net parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Loss, optimizer with weight decay
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # OneCycleLR scheduler - better than ReduceLROnPlateau for convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Mixed precision training for faster GPU utilization
    scaler = torch.cuda.amp.GradScaler()

    # Metrics tracking
    best_val_dice = 0.0
    epochs_no_improve = 0
    early_stop_patience = 10
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    def compute_dice_score(pred, target, threshold=0.5):
        """Compute Dice coefficient."""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-7)
        return dice.item()

    def compute_iou(pred, target, threshold=0.5):
        """Compute IoU (Intersection over Union)."""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-7)
        return iou.item()

    print(f"\n{'='*100}")
    print(f" TRAINING — {EPOCHS} epochs (Save based on Dice Score)")
    print(f"{'='*100}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Dice':>9} | {'Val IoU':>8} | {'Time':>6} | {'VRAM':>6} | {'LR':>10} | Status")
    print("-"*110)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # ========== TRAIN ==========
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step scheduler after each batch (OneCycleLR)

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # ========== VALIDATE ==========
        model.eval()
        val_loss = 0.0
        val_dice_total = 0.0
        val_iou_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)

                # Mixed precision inference
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)

                # Convert logits to probabilities for metrics
                outputs_prob = torch.sigmoid(outputs)

                # Compute metrics for each sample in batch
                for i in range(outputs_prob.size(0)):
                    dice = compute_dice_score(outputs_prob[i], masks[i])
                    iou = compute_iou(outputs_prob[i], masks[i])
                    val_dice_total += dice
                    val_iou_total += iou
                    val_batches += 1

        val_loss /= len(val_dataset)
        val_dice = val_dice_total / val_batches
        val_iou = val_iou_total / val_batches
        elapsed = time.time() - start_time

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # Save best model based on DICE SCORE (not loss)
        status = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_dice': best_val_dice,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'epoch': epoch,
            }, MODEL_SAVE_PATH)
            status = "* BEST"
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                status = "! EARLY STOP"
            else:
                status = f"({epochs_no_improve}/{early_stop_patience})"

        current_lr = optimizer.param_groups[0]['lr']
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**3

        print(f"{epoch:2d}/{EPOCHS:2d}  | {train_loss:10.6f} | {val_loss:10.6f} | "
              f"{val_dice:9.4f} | {val_iou:8.4f} | {elapsed:5.1f}s | {gpu_mem:.2f}GB | LR:{current_lr:.2e} | {status}")

        # Early stopping check
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[!] Early stopping triggered after {epoch} epochs (no improvement for {early_stop_patience} epochs)")
            break

    print(f"\n{'='*80}")
    print(f" TRAINING COMPLETE — Best Val Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
    print(f"{'='*80}")
    print(f"\n[OK] Model saved: {MODEL_SAVE_PATH}")

    # Plot training curves
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    axes[0].plot(epochs_range, history["train_loss"], 'o-', label="Train Loss", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], 'o-', label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss Curves", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice & IoU curves
    axes[1].plot(epochs_range, history["val_dice"], 'o-', label="Val Dice Score", color='green', linewidth=2)
    axes[1].plot(epochs_range, history["val_iou"], 'o-', label="Val IoU", color='blue', linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Segmentation Metrics", fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "thermal_training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[PLOT] Training curves saved: {plot_path}")

    return history

# ============================================================================
# TEST INFERENCE
# ============================================================================

def test_unet_inference():
    """Test U-Net on a sample image."""

    if not MODEL_SAVE_PATH.exists():
        print("Model not found. Train first.")
        return

    # Load model
    model = UNet().to(DEVICE)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n[TEST] Model loaded from: {MODEL_SAVE_PATH}")
    print(f"       Best val loss: {checkpoint['val_loss']:.6f}")

    # Load test image
    test_img_dir = THERMAL_DATASET / 'test' / 'images'
    images = list(test_img_dir.glob('*.jpg'))

    if not images:
        print("No test images found")
        return

    test_path = images[0]
    print(f"       Testing on: {test_path.name}")

    # Load and preprocess
    img = Image.open(test_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.sigmoid(logits)  # Convert logits to probabilities
        pred_mask = pred_mask.squeeze().cpu().numpy()

    # Calculate fault area
    fault_pixels = (pred_mask > 0.5).sum()
    total_pixels = IMG_SIZE * IMG_SIZE
    fault_area_pct = (fault_pixels / total_pixels) * 100

    print(f"\n   Predicted fault area: {fault_area_pct:.1f}%")
    print(f"   Mask range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")

    return pred_mask

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train U-Net on thermal masks')
    parser.add_argument('--test-only', action='store_true', help='Only run test inference')
    args = parser.parse_args()

    if args.test_only:
        test_unet_inference()
    else:
        # Check masks exist
        if not MASKS_DIR.exists():
            print("[ERROR] Masks not found!")
            print("   Run: python generate_thermal_masks.py")
            sys.exit(1)

        # Train
        history = train_unet()

        # Test
        test_unet_inference()

        print("\n[OK] Thermal U-Net training complete!")
        print(f"   Model: {MODEL_SAVE_PATH}")
