# 🌡️ Thermal Fault Detection Module

**U-Net Pixel-Level Segmentation for Thermal Imaging**

---

## Architecture Overview

```
Thermal Image → U-Net Segmentation → Binary Mask → Fault Area % → Risk Assessment
```

**NOT object detection — pure segmentation pipeline**

---

## Module Structure

```
backend/
├── training/
│   ├── train_thermal_yolo.py      # Step 1: YOLO (data prep only)
│   ├── generate_thermal_masks.py  # Step 2: Convert boxes → masks
│   └── train_thermal_unet.py      # Step 3: Train U-Net segmentation
│
├── inference/
│   └── thermal_analyzer.py        # U-Net inference pipeline
│
├── risk_engine/
│   └── severity_analysis.py       # Thermal risk logic (area-based)
│
└── models/
    ├── thermal_yolo.pt            # (used only for mask generation)
    └── thermal_unet.pth           # Main inference model ⭐
```

---

## Training Pipeline

### **Step 1: Prepare Masks (One-Time Setup)**

YOLO is used **only** to convert bounding box annotations to pixel masks:

```bash
cd backend/training

# Generate segmentation masks from YOLO boxes
python generate_thermal_masks.py
```

**Output:** `dataset/Thermal Imaging.v6i.yolov8/masks/`
- train/ (1,676 masks)
- valid/ (masks for validation)
- test/ (masks for testing)

### **Step 2: Train U-Net Segmentation Model**

```bash
python train_thermal_unet.py
```

**Configuration:**
- Architecture: U-Net (encoder-decoder with skip connections)
- Input size: 256×256
- Loss: BCE + Dice combined
- Optimizer: Adam (lr=1e-4)
- Epochs: 20
- Device: CUDA if available

**Output:** `backend/models/thermal_unet.pth`

---

## Inference Pipeline

### Segmentation-Only (NO Classification)

```python
from inference.thermal_analyzer import analyze_thermal_image

result = analyze_thermal_image("thermal_image.jpg", return_base64=True)

# Returns:
{
    "input_type": "thermal",
    "fault_area_percent": 23.5,      # % of pixels classified as fault
    "severity_score": 23.5,          # = fault_area_percent
    "risk_level": "High",            # Based on area thresholds
    "maintenance_suggestion": "...",
    "segmentation_mask_base64": "...",  # PNG overlay (red = fault)
    "segmentation_available": True
}
```

---

## Risk Classification (Area-Based)

**Thermal faults are electrical/insulation issues — area determines severity:**

| Fault Area | Risk Level | Action |
|------------|------------|--------|
| **≥ 30%** | **Critical** | SHUT DOWN SYSTEM - Emergency |
| **15-30%** | **High** | Immediate professional inspection |
| **5-15%** | **Medium** | Schedule inspection soon |
| **< 5%** | **Low** | Routine monitoring |

**No confidence score involved** — segmentation doesn't do classification.

---

## API Usage

### Endpoint: `/analyze`

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@thermal_image.jpg" \
  -F "input_type=thermal"
```

**Response:**
```json
{
  "success": true,
  "input_type": "thermal",
  "analysis": {
    "fault_area_percent": 23.5,
    "fault_area_source": "unet_segmentation",
    "severity_score": 23.5,
    "risk_level": "High",
    "maintenance_suggestion": "URGENT: Significant thermal fault..."
  },
  "alert": {
    "triggered": true,
    "level": "CRITICAL",
    "message": "CRITICAL THERMAL FAULT..."
  },
  "segmentation_mask": "base64..."
}
```

---

## UI Integration

### Thermal Tab Features:

1. **Upload Interface**
   - Drag & drop thermal images
   - Same UI as RGB tab

2. **Segmentation Display**
   - Original thermal image
   - Red overlay mask (fault regions)
   - Color legend: ⚫ Normal | 🟡 Anomaly | 🔴 Critical

3. **Fault Area Visualization**
   - Large percentage display
   - Color-coded bar (green/yellow/red)

4. **Risk Assessment**
   - Risk badge (Low/Medium/High/Critical)
   - Color-coded

5. **Maintenance Suggestion**
   - Area-based recommendations
   - Emergency warnings for >30% coverage

---

## Dataset Structure

```
dataset/Thermal Imaging.v6i.yolov8/
├── data.yaml                    # YOLOv8 config
├── train/
│   ├── images/  (1,676 images)
│   └── labels/  (1,676 YOLO .txt)
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── masks/  (generated)
    ├── train/  (1,676 .png masks)
    ├── valid/
    └── test/
```

**Classes (from YOLO labels - used only for reference):**
1. Connectivity Issue/Loose Connection/Corroded
2. Deteriorated Insulation
3. Faulty Circuit Breakers
4. Normal Operation Condition
5. Overloaded Circuits
6. Phase Imbalance

**Note:** U-Net doesn't classify — it only segments fault vs normal regions.

---

## Key Differences: RGB vs Thermal

| Aspect | RGB Module | Thermal Module |
|--------|------------|----------------|
| **Task** | Classification | Segmentation |
| **Model** | ResNet18 | U-Net |
| **Output** | Class label + confidence | Binary mask |
| **Explainability** | Grad-CAM heatmap | Segmentation overlay |
| **Risk Basis** | Fault type + confidence | Fault area % |
| **Metric** | Accuracy (90.93%) | IoU / Dice score |
| **Inference** | <1 second | <1 second |

---

## Installation

The thermal module requires `ultralytics` (YOLOv8):

```bash
pip install ultralytics
```

All other dependencies are already in `backend/requirements.txt`.

---

## Training Time Estimates

| Task | Time (GPU) | Time (CPU) |
|------|------------|------------|
| **Generate masks** | ~2 minutes | ~2 minutes |
| **Train U-Net** | ~15 minutes | ~2 hours |

**Recommended:** Use Google Colab with T4 GPU (free).

---

## Testing

### Test Thermal Analyzer:

```bash
cd backend
python -m inference.thermal_analyzer
```

### Test via API:

```bash
# Start server
python run_server.py

# In another terminal
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/thermal_image.jpg" \
  -F "input_type=thermal"
```

### Test via UI:

1. Open http://localhost:3000
2. Click "🌡️ Thermal Segmentation" tab
3. Upload thermal image
4. Click "Analyze Thermal Image"
5. View segmentation overlay + fault area %

---

## Model Performance

**U-Net Segmentation Metrics:**
- Dice Score: TBD (after training)
- IoU: TBD
- Precision: TBD
- Recall: TBD

**Inference Speed:** <1 second per image

---

## Visualization Example

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Thermal    │  │  Predicted   │  │   Overlay    │
│   Image      │→ │   Mask       │→ │   (Red =     │
│   (Grayscale)│  │   (Binary)   │  │   Fault)     │
└──────────────┘  └──────────────┘  └──────────────┘
```

Output shows:
- Original thermal scan
- Black/white segmentation mask
- Red overlay on original (fault regions highlighted)

---

## Next Steps

1. ✅ Dataset prepared (masks generated)
2. ⏳ Train U-Net model
3. ⏳ Test inference
4. ⏳ Deploy to production
5. ⏳ Collect real-world thermal images for validation

---

## Integration Status

✅ **Backend:** Thermal analyzer module created
✅ **API:** `/analyze` endpoint supports `input_type=thermal`
✅ **Frontend:** Thermal tab fully functional
✅ **Risk Engine:** Area-based thermal risk logic
⏳ **Model Training:** Awaiting U-Net training

---

**Thermal module ready for training and deployment!**
