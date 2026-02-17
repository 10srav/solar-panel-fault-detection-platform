"""
=============================================================================
 MODEL EVALUATION — evaluate_model.py
 Loads the trained RGB model, evaluates on validation & full dataset,
 generates and SAVES all metrics to backend/training/metrics/ folder.

 Outputs:
   metrics/test_metrics.json          — Full metrics (accuracy, F1, per-class)
   metrics/test_metrics.csv           — Same in CSV format
   metrics/classification_report.txt  — sklearn classification report
   metrics/confusion_matrix.png       — Confusion matrix heatmap
   metrics/per_class_metrics.png      — Per-class precision/recall/F1 bar chart
   metrics/prediction_log.csv         — Every single prediction (image, true, pred, correct)
   metrics/evaluation_summary.txt     — Human-readable summary
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
from torchvision import models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, MODEL_PATH, NUM_CLASSES, CLASS_NAMES,
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, RANDOM_SEED, SPLIT_RATIO,
)
from training.preprocessing import (
    SafeImageFolder, get_val_transform, create_stratified_split,
)
from training.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_class_accuracy_bar,
    save_metrics_json,
    save_metrics_csv,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_trained_model():
    """Load the trained ResNet18 model from checkpoint."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Train the model first with: python -m training.train_rgb_model"
        )

    device = DEVICE
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    train_val_acc = checkpoint.get("val_acc", "N/A")
    train_epoch = checkpoint.get("epoch", "N/A")

    print(f"[MODEL] Loaded: {MODEL_PATH}")
    print(f"[MODEL] Checkpoint val_acc: {train_val_acc}% (epoch {train_epoch})")
    print(f"[MODEL] Device: {device}")

    return model, device, {"checkpoint_val_acc": train_val_acc, "checkpoint_epoch": train_epoch}


# ============================================================================
# EVALUATE ON DATALOADER
# ============================================================================

def evaluate_on_loader(model, data_loader, device, dataset, indices, class_names):
    """
    Run model on every sample in the loader, collect predictions.

    Returns:
        all_labels, all_preds, all_confidences, prediction_details
    """
    all_labels = []
    all_preds = []
    all_confidences = []
    prediction_details = []  # per-image details

    model.eval()
    with torch.no_grad():
        batch_idx = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels_np = labels.numpy()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            preds_np = preds.cpu().numpy()
            confs_np = confidences.cpu().numpy()

            for i in range(len(labels_np)):
                global_idx = batch_idx * data_loader.batch_size + i
                if global_idx < len(indices):
                    sample_idx = indices[global_idx]
                    img_path = dataset.samples[sample_idx][0]
                else:
                    img_path = "unknown"

                true_label = int(labels_np[i])
                pred_label = int(preds_np[i])
                conf = float(confs_np[i])
                correct = true_label == pred_label

                all_labels.append(true_label)
                all_preds.append(pred_label)
                all_confidences.append(conf)

                prediction_details.append({
                    "image": os.path.basename(img_path),
                    "image_path": img_path,
                    "true_class": class_names[true_label],
                    "predicted_class": class_names[pred_label],
                    "correct": correct,
                    "confidence": round(conf, 4),
                })

            batch_idx += 1

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_confidences),
        prediction_details,
    )


# ============================================================================
# SAVE PREDICTION LOG
# ============================================================================

def save_prediction_log(details, save_path):
    """Save every single prediction to CSV for full transparency."""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "true_class", "predicted_class", "correct", "confidence", "image_path"],
        )
        writer.writeheader()
        writer.writerows(details)
    print(f"[LOG] Prediction log saved: {save_path} ({len(details)} samples)")


# ============================================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================================

def save_summary(metrics, checkpoint_info, split_info, misclassified, save_path):
    """Save a human-readable evaluation summary."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" SOLAR PANEL FAULT DETECTION — MODEL EVALUATION REPORT\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL CHECKPOINT INFO\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model file     : {MODEL_PATH}\n")
        f.write(f"  Best epoch     : {checkpoint_info.get('checkpoint_epoch', 'N/A')}\n")
        f.write(f"  Training val_acc: {checkpoint_info.get('checkpoint_val_acc', 'N/A')}%\n\n")

        f.write("DATASET SPLIT\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Dataset path   : {DATASET_PATH}\n")
        f.write(f"  Total images   : {split_info['total']}\n")
        f.write(f"  Training set   : {split_info['train_count']} ({SPLIT_RATIO*100:.0f}%)\n")
        f.write(f"  Validation set : {split_info['val_count']} ({(1-SPLIT_RATIO)*100:.0f}%)\n")
        f.write(f"  Split seed     : {RANDOM_SEED}\n\n")

        f.write("VALIDATION SET RESULTS\n")
        f.write("-" * 40 + "\n")
        acc = metrics["accuracy"] * 100
        total = split_info["val_count"]
        correct = int(round(acc * total / 100))
        wrong = total - correct
        f.write(f"  Accuracy       : {acc:.2f}% ({correct}/{total} correct)\n")
        f.write(f"  Misclassified  : {wrong} samples\n")
        f.write(f"  Precision (macro)  : {metrics['precision_macro']*100:.2f}%\n")
        f.write(f"  Recall (macro)     : {metrics['recall_macro']*100:.2f}%\n")
        f.write(f"  F1-Score (macro)   : {metrics['f1_macro']*100:.2f}%\n")
        f.write(f"  F1-Score (weighted): {metrics['f1_weighted']*100:.2f}%\n\n")

        f.write("PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("  " + "-" * 68 + "\n")
        for cls_name, cls_metrics in metrics.get("per_class", {}).items():
            f.write(
                f"  {cls_name:<30} "
                f"{cls_metrics['precision']*100:>9.2f}% "
                f"{cls_metrics['recall']*100:>9.2f}% "
                f"{cls_metrics['f1_score']*100:>9.2f}% "
                f"{cls_metrics['support']:>10d}\n"
            )
        f.write("\n")

        if misclassified:
            f.write("MISCLASSIFIED SAMPLES\n")
            f.write("-" * 70 + "\n")
            for m in misclassified:
                f.write(
                    f"  {m['image']:<40} "
                    f"True: {m['true_class']:<30} "
                    f"Pred: {m['predicted_class']:<30} "
                    f"Conf: {m['confidence']:.2%}\n"
                )
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(" END OF EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[SUMMARY] Report saved: {save_path}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation():
    """Full evaluation pipeline."""
    print("=" * 70)
    print(" SOLAR PANEL FAULT DETECTION — MODEL EVALUATION")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Load model
    print("\n[1/6] Loading trained model...")
    model, device, checkpoint_info = load_trained_model()

    # Step 2: Load dataset with validation transform (no augmentation)
    print("\n[2/6] Loading dataset...")
    full_dataset = SafeImageFolder(DATASET_PATH)
    val_dataset = SafeImageFolder(DATASET_PATH, transform=get_val_transform())
    class_names = full_dataset.classes
    total_images = len(full_dataset)

    print(f"  Total images: {total_images}")
    for i, name in enumerate(class_names):
        count = sum(1 for t in full_dataset.targets if t == i)
        print(f"  [{i}] {name}: {count}")

    # Step 3: Create same stratified split (deterministic with seed)
    print("\n[3/6] Creating stratified split (seed={})...".format(RANDOM_SEED))
    train_indices, val_indices = create_stratified_split(full_dataset)
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")

    split_info = {
        "total": total_images,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
    }

    # Step 4: Build validation dataloader
    print("\n[4/6] Evaluating on VALIDATION set...")
    from torch.utils.data import Subset, DataLoader

    val_subset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Run evaluation
    y_true, y_pred, y_conf, pred_details = evaluate_on_loader(
        model, val_loader, device, full_dataset, val_indices, class_names,
    )

    # Step 5: Compute metrics
    print("\n[5/6] Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Add extra info
    metrics["evaluation_date"] = datetime.now().isoformat()
    metrics["model_path"] = MODEL_PATH
    metrics["dataset_path"] = DATASET_PATH
    metrics["total_samples_evaluated"] = int(len(y_true))
    metrics["correct_predictions"] = int((y_true == y_pred).sum())
    metrics["incorrect_predictions"] = int((y_true != y_pred).sum())
    metrics["split_seed"] = RANDOM_SEED
    metrics["split_ratio"] = SPLIT_RATIO
    metrics["checkpoint_val_acc"] = checkpoint_info.get("checkpoint_val_acc")
    metrics["checkpoint_epoch"] = checkpoint_info.get("checkpoint_epoch")
    metrics["device"] = str(device)

    # Print results
    acc = metrics["accuracy"] * 100
    correct = metrics["correct_predictions"]
    total = metrics["total_samples_evaluated"]
    print(f"\n  VALIDATION ACCURACY: {acc:.2f}% ({correct}/{total} correct)")
    print(f"  F1-Score (macro):    {metrics['f1_macro']*100:.2f}%")
    print(f"  F1-Score (weighted): {metrics['f1_weighted']*100:.2f}%")

    # sklearn report
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4,
    )
    print(f"\n{report_str}")

    # Misclassified samples
    misclassified = [d for d in pred_details if not d["correct"]]
    if misclassified:
        print(f"\n  MISCLASSIFIED SAMPLES ({len(misclassified)}):")
        for m in misclassified:
            print(f"    {m['image']}: {m['true_class']} -> {m['predicted_class']} (conf={m['confidence']:.2%})")

    # Step 6: Save everything to metrics/ folder
    print(f"\n[6/6] Saving all metrics to {METRICS_DIR}/...")

    # 6a. JSON metrics
    save_metrics_json(metrics, os.path.join(METRICS_DIR, "test_metrics.json"))

    # 6b. CSV metrics
    save_metrics_csv(metrics, os.path.join(METRICS_DIR, "test_metrics.csv"))

    # 6c. Classification report text
    report_path = os.path.join(METRICS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Samples evaluated: {total}\n")
        f.write(f"Accuracy: {acc:.2f}% ({correct}/{total})\n\n")
        f.write(report_str)
    print(f"[REPORT] Classification report saved: {report_path}")

    # 6d. Confusion matrix plot
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(METRICS_DIR, "confusion_matrix.png"),
        title=f"Confusion Matrix — Val Accuracy: {acc:.2f}%",
    )

    # 6e. Per-class metrics bar chart
    plot_class_accuracy_bar(
        metrics,
        save_path=os.path.join(METRICS_DIR, "per_class_metrics.png"),
        title=f"Per-Class Metrics — Val Accuracy: {acc:.2f}%",
    )

    # 6f. Prediction log (every single prediction)
    save_prediction_log(
        pred_details,
        os.path.join(METRICS_DIR, "prediction_log.csv"),
    )

    # 6g. Human-readable summary
    save_summary(
        metrics, checkpoint_info, split_info, misclassified,
        os.path.join(METRICS_DIR, "evaluation_summary.txt"),
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f" EVALUATION COMPLETE — {elapsed:.1f}s")
    print(f" All results saved to: {METRICS_DIR}/")
    print(f"{'=' * 70}")

    # List all saved files
    print("\n  Saved files:")
    for f in sorted(os.listdir(METRICS_DIR)):
        fpath = os.path.join(METRICS_DIR, f)
        size = os.path.getsize(fpath)
        if size > 1024:
            print(f"    {f} ({size/1024:.1f} KB)")
        else:
            print(f"    {f} ({size} bytes)")

    return metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    metrics = run_evaluation()
