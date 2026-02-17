"""
=============================================================================
 THERMAL U-NET MODEL EVALUATION — evaluate_thermal_model.py
 Loads trained U-Net, evaluates on validation & test sets,
 generates and SAVES all segmentation metrics to backend/training/metrics/.

 Outputs:
   metrics/thermal_metrics.json             — Full metrics (Dice, IoU, Precision, Recall, F1, Accuracy)
   metrics/thermal_metrics.csv              — Same in CSV format
   metrics/thermal_classification_report.txt — Human-readable report
   metrics/thermal_confusion_matrix.png     — Pixel-level confusion matrix
   metrics/thermal_per_image_metrics.png    — Per-image Dice/IoU distribution
   metrics/thermal_prediction_log.csv       — Per-image metrics log
   metrics/thermal_sample_predictions.png   — Visual grid of predictions vs ground truth
   metrics/thermal_evaluation_summary.txt   — Full human-readable summary
=============================================================================
"""

import os
import sys
import json
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEVICE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
THERMAL_DATASET = PROJECT_ROOT / "dataset" / "Thermal Imaging.v6i.yolov8"
MASKS_DIR = THERMAL_DATASET / "masks"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "thermal_segmentation_unet_v2.pth"
METRICS_DIR = Path(__file__).parent / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
THRESHOLD = 0.5


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_thermal_model():
    """Load trained U-Net from checkpoint."""
    from training.train_thermal_unet import UNet

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = DEVICE
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ckpt_info = {
        "checkpoint_val_dice": checkpoint.get("val_dice", "N/A"),
        "checkpoint_val_iou": checkpoint.get("val_iou", "N/A"),
        "checkpoint_val_loss": checkpoint.get("val_loss", "N/A"),
        "checkpoint_epoch": checkpoint.get("epoch", "N/A"),
    }

    print(f"[MODEL] Loaded: {MODEL_PATH}")
    print(f"[MODEL] Checkpoint — Dice: {ckpt_info['checkpoint_val_dice']:.4f}, "
          f"IoU: {ckpt_info['checkpoint_val_iou']:.4f}, Epoch: {ckpt_info['checkpoint_epoch']}")
    print(f"[MODEL] Device: {device}")

    return model, device, ckpt_info


# ============================================================================
# METRIC FUNCTIONS (PIXEL-LEVEL)
# ============================================================================

def compute_dice(pred, target, smooth=1e-7):
    """Dice coefficient (2*intersection / union)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def compute_iou(pred, target, smooth=1e-7):
    """Intersection over Union."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_pixel_metrics(pred_binary, target_binary):
    """Compute pixel-level TP, FP, FN, TN, Precision, Recall, F1, Accuracy."""
    pred = pred_binary.flatten().astype(np.float64)
    target = target_binary.flatten().astype(np.float64)

    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    tn = float(((1 - pred) * (1 - target)).sum())

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "specificity": specificity,
    }


# ============================================================================
# EVALUATE ON A SPLIT
# ============================================================================

def evaluate_on_split(model, device, split_name, img_dir, mask_dir):
    """
    Run model on all images in a split, compute per-image and aggregate metrics.

    Returns:
        aggregate_metrics (dict), per_image_results (list of dicts)
    """
    images = sorted(list(img_dir.glob("*.jpg")))
    # Filter: only images that have masks
    images = [img for img in images if (mask_dir / (img.stem + ".png")).exists()]

    print(f"\n  [{split_name.upper()}] Evaluating {len(images)} images...")

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_accuracy = []
    all_specificity = []
    per_image_results = []

    # Aggregate pixel counts
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for i, img_path in enumerate(images):
            mask_path = mask_dir / (img_path.stem + ".png")

            # Load image
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            # Load ground truth mask
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            gt_np = np.array(gt_mask).astype(np.float32) / 255.0
            gt_binary = (gt_np > 0.5).astype(np.float64)

            # Predict
            logits = model(img_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_binary = (pred_prob > THRESHOLD).astype(np.float64)

            # Metrics
            dice = compute_dice(pred_binary, gt_binary)
            iou = compute_iou(pred_binary, gt_binary)
            px = compute_pixel_metrics(pred_binary, gt_binary)

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(px["precision"])
            all_recall.append(px["recall"])
            all_f1.append(px["f1"])
            all_accuracy.append(px["accuracy"])
            all_specificity.append(px["specificity"])

            total_tp += px["tp"]
            total_fp += px["fp"]
            total_fn += px["fn"]
            total_tn += px["tn"]

            fault_area_pred = float(pred_binary.sum() / pred_binary.size * 100)
            fault_area_gt = float(gt_binary.sum() / gt_binary.size * 100)

            per_image_results.append({
                "image": img_path.name,
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "precision": round(px["precision"], 4),
                "recall": round(px["recall"], 4),
                "f1": round(px["f1"], 4),
                "accuracy": round(px["accuracy"], 4),
                "fault_area_pred_pct": round(fault_area_pred, 2),
                "fault_area_gt_pct": round(fault_area_gt, 2),
            })

            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(images)}...")

    # Aggregate (micro-averaged from total pixel counts)
    micro_precision = total_tp / (total_tp + total_fp + 1e-7)
    micro_recall = total_tp / (total_tp + total_fn + 1e-7)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-7)
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-7)

    aggregate = {
        "split": split_name,
        "num_images": len(images),
        # Macro-averaged (per-image mean)
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "dice_median": float(np.median(all_dice)),
        "iou_mean": float(np.mean(all_iou)),
        "iou_std": float(np.std(all_iou)),
        "iou_median": float(np.median(all_iou)),
        "precision_macro": float(np.mean(all_precision)),
        "recall_macro": float(np.mean(all_recall)),
        "f1_macro": float(np.mean(all_f1)),
        "accuracy_macro": float(np.mean(all_accuracy)),
        "specificity_macro": float(np.mean(all_specificity)),
        # Micro-averaged (total pixel counts)
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "accuracy_micro": micro_accuracy,
        # Total pixel counts
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_tn": int(total_tn),
        "total_pixels": int(total_tp + total_fp + total_fn + total_tn),
    }

    return aggregate, per_image_results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_pixel_confusion_matrix(metrics, save_path):
    """Plot pixel-level confusion matrix."""
    tp, fp, fn, tn = metrics["total_tp"], metrics["total_fp"], metrics["total_fn"], metrics["total_tn"]
    cm = np.array([[tn, fp], [fn, tp]])

    # Use percentages for display (total pixels can be huge)
    total = tp + fp + fn + tn
    cm_pct = cm / total * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Background", "Fault"],
        yticklabels=["Background", "Fault"],
        cbar_kws={"label": "Pixel Count"},
    )

    # Add percentage annotations below the counts
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f"({cm_pct[i][j]:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")

    plt.title(f"Pixel-Level Confusion Matrix — {metrics['split'].title()} Set\n"
              f"Dice: {metrics['dice_mean']*100:.2f}% | IoU: {metrics['iou_mean']*100:.2f}%",
              fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrix saved: {save_path}")


def plot_per_image_distribution(per_image_results, save_path):
    """Plot Dice/IoU distribution as histogram + boxplot."""
    dice_scores = [r["dice"] for r in per_image_results]
    iou_scores = [r["iou"] for r in per_image_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Dice histogram
    axes[0][0].hist(dice_scores, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0][0].axvline(np.mean(dice_scores), color="red", linestyle="--", linewidth=2,
                       label=f"Mean: {np.mean(dice_scores):.4f}")
    axes[0][0].set_title("Dice Score Distribution", fontsize=13, fontweight="bold")
    axes[0][0].set_xlabel("Dice Score")
    axes[0][0].set_ylabel("Count")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # IoU histogram
    axes[0][1].hist(iou_scores, bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
    axes[0][1].axvline(np.mean(iou_scores), color="red", linestyle="--", linewidth=2,
                       label=f"Mean: {np.mean(iou_scores):.4f}")
    axes[0][1].set_title("IoU Score Distribution", fontsize=13, fontweight="bold")
    axes[0][1].set_xlabel("IoU Score")
    axes[0][1].set_ylabel("Count")
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # Boxplots
    box_data = [dice_scores, iou_scores]
    bp = axes[1][0].boxplot(box_data, labels=["Dice", "IoU"], patch_artist=True,
                            boxprops=dict(facecolor="#3498db", alpha=0.5))
    bp["boxes"][1].set_facecolor("#2ecc71")
    axes[1][0].set_title("Dice vs IoU — Boxplot", fontsize=13, fontweight="bold")
    axes[1][0].set_ylabel("Score")
    axes[1][0].grid(True, alpha=0.3)

    # Scatter: Dice vs IoU per image
    axes[1][1].scatter(dice_scores, iou_scores, alpha=0.6, color="#9b59b6", edgecolors="white")
    axes[1][1].plot([0, 1], [0, 1], "r--", alpha=0.5)
    axes[1][1].set_title("Dice vs IoU per Image", fontsize=13, fontweight="bold")
    axes[1][1].set_xlabel("Dice Score")
    axes[1][1].set_ylabel("IoU Score")
    axes[1][1].grid(True, alpha=0.3)

    plt.suptitle(f"Per-Image Segmentation Metrics (n={len(per_image_results)})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Per-image distribution saved: {save_path}")


def plot_metrics_bar_chart(metrics, save_path):
    """Bar chart of all aggregate metrics."""
    metric_names = ["Dice", "IoU", "Precision", "Recall", "F1", "Accuracy", "Specificity"]
    macro_vals = [
        metrics["dice_mean"] * 100,
        metrics["iou_mean"] * 100,
        metrics["precision_macro"] * 100,
        metrics["recall_macro"] * 100,
        metrics["f1_macro"] * 100,
        metrics["accuracy_macro"] * 100,
        metrics["specificity_macro"] * 100,
    ]
    micro_vals = [
        metrics["dice_mean"] * 100,  # Dice is inherently per-image
        metrics["iou_mean"] * 100,
        metrics["precision_micro"] * 100,
        metrics["recall_micro"] * 100,
        metrics["f1_micro"] * 100,
        metrics["accuracy_micro"] * 100,
        metrics["accuracy_micro"] * 100,  # placeholder
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width / 2, macro_vals, width, label="Macro (per-image avg)", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, micro_vals, width, label="Micro (pixel-level)", color="#e74c3c", alpha=0.8)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Thermal U-Net Segmentation Metrics — {metrics['split'].title()} Set",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim([0, 105])

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Metrics bar chart saved: {save_path}")


def plot_sample_predictions(model, device, img_dir, mask_dir, save_path, n_samples=8):
    """Plot grid: original | ground truth | predicted mask | overlay."""
    images = sorted(list(img_dir.glob("*.jpg")))
    images = [img for img in images if (mask_dir / (img.stem + ".png")).exists()]

    # Pick evenly spaced samples
    if len(images) > n_samples:
        step = len(images) // n_samples
        images = images[::step][:n_samples]

    fig, axes = plt.subplots(len(images), 4, figsize=(16, 4 * len(images)))
    if len(images) == 1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for idx, img_path in enumerate(images):
            mask_path = mask_dir / (img_path.stem + ".png")

            # Load
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_np = np.array(img_resized).astype(np.float32) / 255.0

            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            gt_np = np.array(gt_mask).astype(np.float32) / 255.0

            # Predict
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(img_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_binary = (pred_prob > THRESHOLD).astype(np.float32)

            # Dice for this image
            dice = compute_dice(pred_binary, (gt_np > 0.5).astype(np.float64))

            # Overlay
            overlay = img_np.copy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_binary * 0.3, 0, 1)

            # Plot
            axes[idx][0].imshow(img_np)
            axes[idx][0].set_title(f"Input: {img_path.name[:25]}", fontsize=9)
            axes[idx][0].axis("off")

            axes[idx][1].imshow(gt_np, cmap="hot", vmin=0, vmax=1)
            axes[idx][1].set_title("Ground Truth", fontsize=9)
            axes[idx][1].axis("off")

            axes[idx][2].imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
            axes[idx][2].set_title(f"Prediction (Dice={dice:.3f})", fontsize=9)
            axes[idx][2].axis("off")

            axes[idx][3].imshow(overlay)
            axes[idx][3].set_title("Overlay", fontsize=9)
            axes[idx][3].axis("off")

    plt.suptitle("Thermal U-Net: Sample Predictions vs Ground Truth",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Sample predictions saved: {save_path}")


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_prediction_log(per_image_results, save_path):
    """Save per-image metrics to CSV."""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_results[0].keys())
        writer.writeheader()
        writer.writerows(per_image_results)
    print(f"[LOG] Prediction log saved: {save_path} ({len(per_image_results)} images)")


def save_metrics_json(metrics, save_path):
    """Save metrics to JSON."""
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[METRICS] JSON saved: {save_path}")


def save_metrics_csv(metrics, save_path):
    """Save aggregate metrics to CSV."""
    rows = [["Metric", "Value"]]
    skip_keys = {"split", "total_tp", "total_fp", "total_fn", "total_tn", "total_pixels"}
    for k, v in metrics.items():
        if k in skip_keys:
            continue
        if isinstance(v, float):
            rows.append([k, f"{v:.6f}"])
        else:
            rows.append([k, str(v)])
    # Add pixel counts
    rows.append([])
    rows.append(["Pixel Counts", ""])
    rows.append(["True Positives", str(metrics["total_tp"])])
    rows.append(["False Positives", str(metrics["total_fp"])])
    rows.append(["False Negatives", str(metrics["total_fn"])])
    rows.append(["True Negatives", str(metrics["total_tn"])])
    rows.append(["Total Pixels", str(metrics["total_pixels"])])

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[METRICS] CSV saved: {save_path}")


def save_summary(val_metrics, test_metrics, ckpt_info, save_path):
    """Save human-readable evaluation summary."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" THERMAL U-NET SEGMENTATION — MODEL EVALUATION REPORT\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL CHECKPOINT INFO\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model file       : {MODEL_PATH}\n")
        f.write(f"  Architecture     : U-Net (3-channel input, 1-channel output)\n")
        f.write(f"  Task             : Binary Segmentation (Fault vs Background)\n")
        f.write(f"  Best epoch       : {ckpt_info.get('checkpoint_epoch', 'N/A')}\n")
        f.write(f"  Checkpoint Dice  : {ckpt_info.get('checkpoint_val_dice', 0):.4f} "
                f"({ckpt_info.get('checkpoint_val_dice', 0)*100:.2f}%)\n")
        f.write(f"  Checkpoint IoU   : {ckpt_info.get('checkpoint_val_iou', 0):.4f} "
                f"({ckpt_info.get('checkpoint_val_iou', 0)*100:.2f}%)\n\n")

        for metrics, label in [(val_metrics, "VALIDATION"), (test_metrics, "TEST")]:
            if metrics is None:
                continue
            f.write(f"{label} SET RESULTS ({metrics['num_images']} images)\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Dice Score (mean ± std) : {metrics['dice_mean']*100:.2f}% ± {metrics['dice_std']*100:.2f}%\n")
            f.write(f"  Dice Score (median)     : {metrics['dice_median']*100:.2f}%\n")
            f.write(f"  IoU (mean ± std)        : {metrics['iou_mean']*100:.2f}% ± {metrics['iou_std']*100:.2f}%\n")
            f.write(f"  IoU (median)            : {metrics['iou_median']*100:.2f}%\n")
            f.write(f"\n  Pixel-Level Metrics (Macro — per-image average):\n")
            f.write(f"    Precision     : {metrics['precision_macro']*100:.2f}%\n")
            f.write(f"    Recall        : {metrics['recall_macro']*100:.2f}%\n")
            f.write(f"    F1-Score      : {metrics['f1_macro']*100:.2f}%\n")
            f.write(f"    Accuracy      : {metrics['accuracy_macro']*100:.2f}%\n")
            f.write(f"    Specificity   : {metrics['specificity_macro']*100:.2f}%\n")
            f.write(f"\n  Pixel-Level Metrics (Micro — total pixel counts):\n")
            f.write(f"    Precision     : {metrics['precision_micro']*100:.2f}%\n")
            f.write(f"    Recall        : {metrics['recall_micro']*100:.2f}%\n")
            f.write(f"    F1-Score      : {metrics['f1_micro']*100:.2f}%\n")
            f.write(f"    Accuracy      : {metrics['accuracy_micro']*100:.2f}%\n")
            f.write(f"\n  Pixel Counts:\n")
            f.write(f"    True Positives  : {metrics['total_tp']:>12,}\n")
            f.write(f"    False Positives : {metrics['total_fp']:>12,}\n")
            f.write(f"    False Negatives : {metrics['total_fn']:>12,}\n")
            f.write(f"    True Negatives  : {metrics['total_tn']:>12,}\n")
            f.write(f"    Total Pixels    : {metrics['total_pixels']:>12,}\n\n")

        f.write("=" * 70 + "\n")
        f.write(" END OF THERMAL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[SUMMARY] Report saved: {save_path}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation():
    """Full thermal model evaluation pipeline."""
    print("=" * 70)
    print(" THERMAL U-NET SEGMENTATION — MODEL EVALUATION")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Load model
    print("\n[1/7] Loading trained U-Net model...")
    model, device, ckpt_info = load_thermal_model()

    # Step 2: Evaluate on VALIDATION set
    print("\n[2/7] Evaluating on VALIDATION set...")
    val_img_dir = THERMAL_DATASET / "valid" / "images"
    val_mask_dir = MASKS_DIR / "valid"
    val_metrics, val_details = evaluate_on_split(model, device, "validation", val_img_dir, val_mask_dir)

    print(f"\n  VALIDATION RESULTS:")
    print(f"    Dice (mean):      {val_metrics['dice_mean']*100:.2f}%")
    print(f"    IoU (mean):       {val_metrics['iou_mean']*100:.2f}%")
    print(f"    Precision (macro):{val_metrics['precision_macro']*100:.2f}%")
    print(f"    Recall (macro):   {val_metrics['recall_macro']*100:.2f}%")
    print(f"    F1 (macro):       {val_metrics['f1_macro']*100:.2f}%")
    print(f"    Accuracy (macro): {val_metrics['accuracy_macro']*100:.2f}%")

    # Step 3: Evaluate on TEST set
    print("\n[3/7] Evaluating on TEST set...")
    test_img_dir = THERMAL_DATASET / "test" / "images"
    test_mask_dir = MASKS_DIR / "test"
    test_metrics, test_details = evaluate_on_split(model, device, "test", test_img_dir, test_mask_dir)

    print(f"\n  TEST RESULTS:")
    print(f"    Dice (mean):      {test_metrics['dice_mean']*100:.2f}%")
    print(f"    IoU (mean):       {test_metrics['iou_mean']*100:.2f}%")
    print(f"    Precision (macro):{test_metrics['precision_macro']*100:.2f}%")
    print(f"    Recall (macro):   {test_metrics['recall_macro']*100:.2f}%")
    print(f"    F1 (macro):       {test_metrics['f1_macro']*100:.2f}%")
    print(f"    Accuracy (macro): {test_metrics['accuracy_macro']*100:.2f}%")

    # Step 4: Save JSON and CSV
    print(f"\n[4/7] Saving metrics to {METRICS_DIR}/...")
    combined_metrics = {
        "evaluation_date": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH),
        "threshold": THRESHOLD,
        "img_size": IMG_SIZE,
        "device": str(device),
        "checkpoint": ckpt_info,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_metrics_json(combined_metrics, METRICS_DIR / "thermal_metrics.json")
    save_metrics_csv(val_metrics, METRICS_DIR / "thermal_metrics_validation.csv")
    save_metrics_csv(test_metrics, METRICS_DIR / "thermal_metrics_test.csv")

    # Step 5: Save prediction logs
    print("\n[5/7] Saving prediction logs...")
    save_prediction_log(val_details, METRICS_DIR / "thermal_prediction_log_validation.csv")
    save_prediction_log(test_details, METRICS_DIR / "thermal_prediction_log_test.csv")

    # Step 6: Generate plots
    print("\n[6/7] Generating plots...")
    # Confusion matrices
    plot_pixel_confusion_matrix(val_metrics, METRICS_DIR / "thermal_confusion_matrix_validation.png")
    plot_pixel_confusion_matrix(test_metrics, METRICS_DIR / "thermal_confusion_matrix_test.png")

    # Per-image distribution (validation)
    plot_per_image_distribution(val_details, METRICS_DIR / "thermal_per_image_metrics_validation.png")
    plot_per_image_distribution(test_details, METRICS_DIR / "thermal_per_image_metrics_test.png")

    # Metrics bar charts
    plot_metrics_bar_chart(val_metrics, METRICS_DIR / "thermal_metrics_bar_validation.png")
    plot_metrics_bar_chart(test_metrics, METRICS_DIR / "thermal_metrics_bar_test.png")

    # Sample predictions
    plot_sample_predictions(model, device, val_img_dir, val_mask_dir,
                           METRICS_DIR / "thermal_sample_predictions_validation.png", n_samples=6)
    plot_sample_predictions(model, device, test_img_dir, test_mask_dir,
                           METRICS_DIR / "thermal_sample_predictions_test.png", n_samples=6)

    # Step 7: Save human-readable summary
    print("\n[7/7] Saving evaluation summary...")
    save_summary(val_metrics, test_metrics, ckpt_info,
                 METRICS_DIR / "thermal_evaluation_summary.txt")

    # Also save classification report text
    report_path = METRICS_DIR / "thermal_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Thermal U-Net Segmentation — Classification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")

        for metrics, label in [(val_metrics, "VALIDATION"), (test_metrics, "TEST")]:
            f.write(f"{'='*50}\n {label} SET ({metrics['num_images']} images)\n{'='*50}\n\n")
            f.write(f"  {'Metric':<25} {'Macro':>10} {'Micro':>10}\n")
            f.write(f"  {'-'*45}\n")
            f.write(f"  {'Dice Score':<25} {metrics['dice_mean']*100:>9.2f}%       —\n")
            f.write(f"  {'IoU':<25} {metrics['iou_mean']*100:>9.2f}%       —\n")
            f.write(f"  {'Precision':<25} {metrics['precision_macro']*100:>9.2f}% {metrics['precision_micro']*100:>9.2f}%\n")
            f.write(f"  {'Recall':<25} {metrics['recall_macro']*100:>9.2f}% {metrics['recall_micro']*100:>9.2f}%\n")
            f.write(f"  {'F1-Score':<25} {metrics['f1_macro']*100:>9.2f}% {metrics['f1_micro']*100:>9.2f}%\n")
            f.write(f"  {'Pixel Accuracy':<25} {metrics['accuracy_macro']*100:>9.2f}% {metrics['accuracy_micro']*100:>9.2f}%\n\n")

    print(f"[REPORT] Classification report saved: {report_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f" THERMAL EVALUATION COMPLETE — {elapsed:.1f}s")
    print(f" All results saved to: {METRICS_DIR}/")
    print(f"{'=' * 70}")

    # List thermal files
    print("\n  Thermal metric files saved:")
    for f in sorted(os.listdir(METRICS_DIR)):
        if f.startswith("thermal_"):
            fpath = os.path.join(METRICS_DIR, f)
            size = os.path.getsize(fpath)
            if size > 1024:
                print(f"    {f} ({size/1024:.1f} KB)")
            else:
                print(f"    {f} ({size} bytes)")

    return combined_metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    metrics = run_evaluation()
