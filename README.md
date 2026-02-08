# 🌞 Solar Panel Fault Detection AI System

**Production-ready deep learning system for automated solar panel fault detection and diagnosis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Backend API](#backend-api)
- [Frontend Dashboard](#frontend-dashboard)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This system uses **deep learning** (ResNet18 with Grad-CAM) to automatically detect and classify faults in solar panel RGB images. It provides:

- **6-class fault detection**: Bird droppings, Clean, Dusty, Electrical damage, Physical damage, Snow covered
- **93.38% validation accuracy** (98.3% on test samples)
- **Real-time inference** with visual explanations (Grad-CAM heatmaps)
- **Risk assessment** engine for maintenance prioritization
- **Production-ready REST API** with FastAPI
- **Modern web dashboard** for non-technical users

### 🎓 Use Cases

- ✅ Solar plant maintenance automation
- ✅ Predictive maintenance scheduling
- ✅ Quality control in manufacturing
- ✅ Academic research & education
- ✅ Real-time monitoring systems

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              React Dashboard / API Clients                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Inference   │  │  Grad-CAM    │  │ Risk Engine  │         │
│  │  Pipeline    │  │ Explainability│  │  Severity   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING MODEL                          │
│         ResNet18 (Transfer Learning + Fine-tuning)              │
│              Trained on RGB Solar Panel Dataset                 │
└─────────────────────────────────────────────────────────────────┘
```

### 🔬 Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | PyTorch 2.0+ |
| **Architecture** | ResNet18 (pre-trained on ImageNet) |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React (HTML/CSS/JS) |
| **Data Augmentation** | Torchvision transforms |
| **Metrics** | Precision, Recall, F1-Score, Confusion Matrix |
| **Training** | Cross-Entropy Loss + Adam Optimizer + StepLR Scheduler |

---

## ✨ Features

### 🤖 Machine Learning

- ✅ **Transfer learning** from ImageNet-pretrained ResNet18
- ✅ **Layer freezing** strategy for efficient training
- ✅ **Class-weighted loss** to handle imbalanced data
- ✅ **Early stopping** to prevent overfitting
- ✅ **Learning rate scheduling** for optimal convergence

### 📊 Explainability & Trust

- ✅ **Grad-CAM visualization** - See where the model looks
- ✅ **Confidence scores** - Know when to trust predictions
- ✅ **Fault area percentage** - Quantify damage extent
- ✅ **Risk level classification** - Low, Medium, High

### 🚀 Production Features

- ✅ **REST API** with automatic documentation (Swagger/OpenAPI)
- ✅ **Image validation** - File type, size, corruption checks
- ✅ **Error handling** - Graceful failures with informative messages
- ✅ **CORS support** - Easy frontend integration
- ✅ **Health check endpoint** - Monitor system status

### 📈 Experiment Tracking

- ✅ **Hyperparameter logging** - Reproducible experiments
- ✅ **Epoch-wise metrics** - Track training progress
- ✅ **Confusion matrices** - Understand model errors
- ✅ **Training curves** - Visualize learning
- ✅ **JSON/CSV exports** - Easy analysis

---

## 📦 Installation

### Prerequisites

- **Python 3.10 or higher**
- **CUDA-capable GPU** (optional, but recommended for training)
- **8GB+ RAM**
- **Windows / Linux / macOS**

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/solar-panel-fault-ai.git
cd solar-panel-fault-ai
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd solar_ai_system
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
CUDA available: True  # or False if CPU-only
```

---

## 🚀 Quick Start

### Option A: Use Pre-trained Model (Recommended)

The system comes with a pre-trained model (`models/rgb_fault_model.pth`).

**1. Start the API server:**

```bash
cd solar_ai_system
python run_server.py
```

**2. Open your browser:**

Navigate to: `http://localhost:8000`

**3. Upload an image** and get instant predictions!

---

### Option B: Train From Scratch

If you want to retrain the model with your own data:

**1. Prepare your dataset** (see [Training](#training) section)

**2. Run training:**

```bash
cd solar_ai_system
python run_training.py
```

Training will take ~35 minutes (RTX 3050) for 15 epochs.

---

## 🎓 Training

### Dataset Structure

The dataset must be organized in the following format:

```
dataset/
└── PRoject/
    ├── Bird_drop_generateds/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    ├── Clean/
    ├── Dusty/
    ├── Electrical_damage_generated/
    ├── Physcial_damage_generated/
    └── Snow_covered_generated/
```

**Current dataset:** 7,547 images across 6 classes

### Training Configuration

Edit `solar_ai_system/config.py` to modify hyperparameters:

```python
# Hyperparameters
LEARNING_RATE = 1e-4       # Initial learning rate
EPOCHS = 15                # Maximum training epochs
BATCH_SIZE = 32            # Batch size
SPLIT_RATIO = 0.8          # Train/val split (80/20)
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs

# Learning rate schedule
STEP_SIZE = 5              # Reduce LR every N epochs
GAMMA = 0.5                # LR reduction factor
```

### Run Training

```bash
python run_training.py
```

### Training Output

The training script will:

1. ✅ Load and preprocess the dataset
2. ✅ Create train/validation splits (stratified)
3. ✅ Train the model with progress logging
4. ✅ Save the best model automatically
5. ✅ Generate visualizations:
   - Confusion matrix (`plots/confusion_matrix.png`)
   - Training curves (`plots/training_curves.png`)
   - Per-class accuracy chart (`plots/class_accuracy.png`)
6. ✅ Save metrics:
   - JSON format (`training/logs/metrics.json`)
   - CSV format (`training/logs/metrics.csv`)
   - Experiment log (`training/logs/experiment_TIMESTAMP.json`)

### Expected Training Time

| Hardware | Time per Epoch | Total Time (15 epochs) |
|----------|----------------|------------------------|
| RTX 3050 Laptop | ~140 seconds | ~35 minutes |
| RTX 3060 | ~90 seconds | ~23 minutes |
| RTX 3090 | ~50 seconds | ~13 minutes |
| CPU (i7-12700) | ~600 seconds | ~2.5 hours |

---

## 🌐 Backend API

### Start the Server

```bash
cd solar_ai_system
python run_server.py
```

Server starts at: **http://localhost:8000**

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend or API info |
| `/health` | GET | Health check & model status |
| `/classes` | GET | List available fault classes |
| `/analyze` | POST | Upload image for fault detection |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Example: Analyze Image

**Using cURL:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@panel_image.jpg"
```

**Using Python:**

```python
import requests

url = "http://localhost:8000/analyze"
files = {'file': open('panel_image.jpg', 'rb')}
response = requests.post(url, files=files)

result = response.json()
print(f"Fault: {result['prediction']['class_name']}")
print(f"Confidence: {result['prediction']['confidence']:.1f}%")
print(f"Risk Level: {result['analysis']['risk_level']}")
```

### Response Format

```json
{
  "success": true,
  "filename": "panel_image.jpg",
  "prediction": {
    "class_index": 2,
    "class_name": "Dusty",
    "confidence": 95.3
  },
  "analysis": {
    "fault_area_percent": 12.5,
    "severity_score": 35.8,
    "risk_level": "Medium",
    "maintenance_suggestion": "Surface cleaning — remove dust accumulation for optimal output"
  },
  "alert": {
    "level": "warning",
    "message": "Moderate fault detected"
  },
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

---

## 🎨 Frontend Dashboard

### Overview

The system provides a **modern, interactive web dashboard** designed for solar plant operators and maintenance engineers.

### 🔹 Core Design Goals

- ✅ Simple for non-technical users
- ✅ Visual, not text-heavy
- ✅ Fast decision support
- ✅ Real-time fault insight

### 🔹 Main UI Components

#### 1️⃣ **Image Upload Panel**

- Drag & drop or browse option
- Accepts RGB solar panel images
- Validates file type and size
- **Entry point for fault analysis**

#### 2️⃣ **Prediction Summary Card**

Displays key results clearly:
- **Detected Fault Type**
- **Confidence (%)**
- **Status Badge** (Normal / Fault Detected)

**Purpose:** Gives instant understanding of panel condition.

#### 3️⃣ **Grad-CAM Visualization Area**

- Shows original image
- Overlays heatmap highlighting where the model focused
- **Provides explainability** (WHY the model predicted fault)

#### 4️⃣ **Fault Severity & Risk Panel**

- **Fault Area %**
- **Severity Score**
- **Risk Level** (Low / Medium / High)
- Color-coded indicators:
  - 🟢 **Low** - Minor issue, routine maintenance
  - 🟡 **Medium** - Moderate issue, schedule maintenance soon
  - 🔴 **High** - Critical issue, immediate action required

**Purpose:** Converts AI output into maintenance priority.

#### 5️⃣ **Maintenance Suggestion Box**

Clear, actionable recommendations:
- "Cleaning required — remove bird droppings from panel surface"
- "Panel replacement — physical damage detected, replace panel"
- "No action needed — panel is in good condition"

**Purpose:** Direct guidance for maintenance teams.

### Access the Dashboard

1. Start the backend: `python run_server.py`
2. Open browser: `http://localhost:8000`
3. Upload an image and analyze!

---

## 📁 Project Structure

```
solar_panel/
│
├── solar_ai_system/              ← Main system directory
│   │
│   ├── training/                 ← MODEL DEVELOPMENT ZONE
│   │   ├── train_rgb_model.py    ← Training script
│   │   ├── preprocessing.py      ← Data pipeline
│   │   ├── metrics.py            ← Evaluation metrics & plots
│   │   ├── experiment_logger.py  ← Experiment tracking
│   │   ├── __init__.py
│   │   ├── plots/                ← Auto-saved visualizations
│   │   │   ├── confusion_matrix.png
│   │   │   ├── training_curves.png
│   │   │   └── class_accuracy.png
│   │   └── logs/                 ← Experiment logs (JSON/CSV)
│   │       ├── experiment_TIMESTAMP.json
│   │       ├── metrics.json
│   │       ├── metrics.csv
│   │       └── epoch_logs.csv
│   │
│   ├── models/                   ← SAVED MODEL WEIGHTS
│   │   ├── rgb_fault_model.pth   ← Trained model (43MB)
│   │   └── class_mapping.json    ← Class index mapping
│   │
│   ├── explainability/           ← Grad-CAM engine
│   │   ├── gradcam_engine.py
│   │   └── __init__.py
│   │
│   ├── risk_engine/              ← Severity analysis
│   │   ├── severity_analysis.py
│   │   └── __init__.py
│   │
│   ├── inference/                ← Prediction pipeline
│   │   ├── predictor.py
│   │   └── __init__.py
│   │
│   ├── backend/                  ← FastAPI server
│   │   ├── server.py             ← API endpoints
│   │   └── __init__.py
│   │
│   ├── frontend/                 ← React dashboard
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   │
│   ├── notebooks/                ← Jupyter notebooks
│   │   ├── 01_data_exploration.ipynb
│   │   └── 02_model_analysis.ipynb
│   │
│   ├── unwanted/                 ← Archived/deprecated files
│   │
│   ├── config.py                 ← Central configuration
│   ├── requirements.txt          ← Dependencies
│   ├── run_training.py           ← Training launcher
│   ├── run_server.py             ← API server launcher
│   └── run_inference.py          ← Standalone inference
│
├── dataset/                      ← RGB dataset (7,547 images)
│   └── PRoject/
│       ├── Bird_drop_generateds/
│       ├── Clean/
│       ├── Dusty/
│       ├── Electrical_damage_generated/
│       ├── Physcial_damage_generated/
│       └── Snow_covered_generated/
│
├── test_api.py                   ← API testing script
└── README.md                     ← This file
```

---

## 📊 Model Performance

### Validation Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **93.38%** |
| **Precision (Macro)** | 94.1% |
| **Recall (Macro)** | 93.9% |
| **F1-Score (Macro)** | 94.0% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bird_drop_generateds | 93.8% | 94.4% | 94.1% | 288 |
| Clean | 88.1% | 92.6% | 90.3% | 336 |
| Dusty | 84.0% | 77.6% | 80.7% | 250 |
| Electrical_damage_generated | **100.0%** | **100.0%** | **100.0%** | 198 |
| Physcial_damage_generated | 99.6% | 98.7% | 99.1% | 226 |
| Snow_covered_generated | 99.1% | **100.0%** | 99.5% | 212 |

### Real-World Testing

**Test:** 60 random samples (10 per class)
**Result:** **98.3% accuracy** (59/60 correct)

Only 1 misclassification: Dusty panel → Bird droppings (visually similar)

---

## 📖 API Documentation

### Interactive Docs

Visit **http://localhost:8000/docs** for:
- Interactive API testing (Swagger UI)
- Request/response schemas
- Try-it-out functionality

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": [
    "Bird_drop_generateds",
    "Clean",
    "Dusty",
    "Electrical_damage_generated",
    "Physcial_damage_generated",
    "Snow_covered_generated"
  ]
}
```

### Analyze Image

**POST** `/analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Supported formats:** JPG, JPEG, PNG, BMP, TIFF, WEBP
**Max file size:** 20 MB

**Response:** See [Response Format](#response-format) above

---

## 🔧 Troubleshooting

### Issue: Model not found error

**Error:** `FileNotFoundError: Model not found`

**Solution:**
1. Check if `models/rgb_fault_model.pth` exists
2. If missing, run training: `python run_training.py`
3. Or download pre-trained model from releases

---

### Issue: CUDA out of memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Reduce batch size in `config.py`: `BATCH_SIZE = 16` or `8`
2. Use CPU: Set `DEVICE = "cpu"` in `config.py`
3. Close other GPU-intensive applications

---

### Issue: Import errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -r solar_ai_system/requirements.txt
```

---

### Issue: Port already in use

**Error:** `Address already in use`

**Solution:**
1. Change port in `config.py`: `API_PORT = 8001`
2. Or kill process using port 8000:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F

   # Linux/macOS
   lsof -ti:8000 | xargs kill -9
   ```

---

### Issue: Poor model performance

**Symptoms:** Low accuracy, wrong predictions

**Solutions:**
1. **More training epochs:** Increase `EPOCHS` in `config.py`
2. **Better data augmentation:** Edit `preprocessing.py`
3. **Stronger backbone:** Switch to ResNet34/50 in `train_rgb_model.py`
4. **More data:** Add more training images
5. **Check data quality:** Remove corrupted/mislabeled images

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r solar_ai_system/requirements.txt
pip install pytest black

# Run tests
pytest tests/

# Format code
black solar_ai_system/
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Project Maintainer:** Your Name
**Email:** your.email@example.com
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **FastAPI** for the modern, fast web framework
- **ResNet Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Grad-CAM Paper:** [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

---

## 🎯 Roadmap

- [ ] Add thermal imaging support (IR camera)
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Mobile app (iOS/Android)
- [ ] Real-time video processing
- [ ] Multi-language support
- [ ] Database integration for history tracking
- [ ] Advanced analytics dashboard
- [ ] Automated report generation (PDF)

---

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{solar_fault_detection_2024,
  author = {Your Name},
  title = {Solar Panel Fault Detection AI System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/solar-panel-fault-ai}
}
```

---

<p align="center">
  <strong>Built with ❤️ for sustainable energy monitoring</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made with Python">
  <img src="https://img.shields.io/badge/Powered%20by-PyTorch-ee4c2c.svg" alt="Powered by PyTorch">
  <img src="https://img.shields.io/badge/Deployable-FastAPI-009688.svg" alt="Deployable with FastAPI">
</p>
