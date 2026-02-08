# 👨‍💻 Developer Guide - Solar Panel Fault Detection AI

**Complete technical documentation for developers and contributors**

---

## 📚 Table of Contents

1. [Development Setup](#development-setup)
2. [Code Architecture](#code-architecture)
3. [Adding New Features](#adding-new-features)
4. [Training Custom Models](#training-custom-models)
5. [API Extension](#api-extension)
6. [Testing](#testing)
7. [Best Practices](#best-practices)
8. [Deployment](#deployment)

---

## 🛠️ Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/solar-panel-fault-ai.git
cd solar-panel-fault-ai

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
cd solar_ai_system
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy jupyter
```

### 2. Verify Installation

```bash
# Check PyTorch + CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check FastAPI
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

### 3. IDE Setup

**Recommended IDEs:**
- **VS Code** with Python extension
- **PyCharm Professional**
- **Jupyter Lab** for notebooks

**VS Code Extensions:**
- Python (Microsoft)
- Pylance
- Black Formatter
- Git Graph

---

## 🏗️ Code Architecture

### Module Overview

```
solar_ai_system/
│
├── config.py                    ← SINGLE SOURCE OF TRUTH
│                                   All hyperparameters, paths, constants
│
├── training/                    ← TRAINING PIPELINE
│   ├── train_rgb_model.py       ← Main training logic
│   ├── preprocessing.py         ← Data loading & augmentation
│   ├── metrics.py               ← Evaluation metrics & plotting
│   ├── experiment_logger.py     ← Experiment tracking
│   └── __init__.py
│
├── inference/                   ← INFERENCE PIPELINE
│   ├── predictor.py             ← Load model & predict
│   └── __init__.py
│
├── explainability/              ← GRAD-CAM
│   ├── gradcam_engine.py        ← Generate heatmaps
│   └── __init__.py
│
├── risk_engine/                 ← BUSINESS LOGIC
│   ├── severity_analysis.py    ← Risk assessment
│   └── __init__.py
│
└── backend/                     ← WEB API
    ├── server.py                ← FastAPI routes
    └── __init__.py
```

### Design Principles

1. **Separation of Concerns:** Each module has a single responsibility
2. **Configuration-Driven:** All parameters in `config.py`
3. **Type Hints:** Use Python type annotations
4. **Docstrings:** Google-style docstrings for all functions
5. **Error Handling:** Graceful failures with informative messages

---

## ➕ Adding New Features

### Example: Add a New Fault Class

**Step 1: Update Dataset**

```
dataset/PRoject/
└── New_Fault_Class/      ← Add new folder
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

**Step 2: Update Config**

```python
# config.py
CLASS_NAMES = [
    "Bird_drop_generateds",
    "Clean",
    "Dusty",
    "Electrical_damage_generated",
    "Physcial_damage_generated",
    "Snow_covered_generated",
    "New_Fault_Class",        # ← Add new class
]

NUM_CLASSES = len(CLASS_NAMES)  # Auto-updates to 7
```

**Step 3: Add Maintenance Suggestion**

```python
# config.py
MAINTENANCE_SUGGESTIONS = {
    "New_Fault_Class": "Your maintenance suggestion here",
    # ... existing suggestions
}
```

**Step 4: Retrain Model**

```bash
python run_training.py
```

Done! The system automatically adapts to the new class.

---

### Example: Add New Metric

**Step 1: Implement in metrics.py**

```python
# training/metrics.py

def compute_top_k_accuracy(y_true, y_pred_probs, k=3):
    """
    Compute top-k accuracy.

    Args:
        y_true: True labels (numpy array)
        y_pred_probs: Prediction probabilities (numpy array, shape: [N, num_classes])
        k: Top-k value

    Returns:
        float: Top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(np.mean(correct))
```

**Step 2: Use in Training**

```python
# training/train_rgb_model.py

from training.metrics import compute_top_k_accuracy

# After getting predictions
top3_acc = compute_top_k_accuracy(y_true, y_pred_probs, k=3)
print(f"Top-3 Accuracy: {top3_acc:.2f}%")
```

---

## 🎓 Training Custom Models

### Switch Backbone Architecture

**Option 1: ResNet34**

```python
# training/train_rgb_model.py

import torchvision.models as models

def build_model():
    # Replace this line:
    # model = models.resnet18(pretrained=True)

    # With:
    model = models.resnet34(pretrained=True)

    # Rest stays the same
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    # ...
```

**Option 2: EfficientNet**

```python
import torchvision.models as models

def build_model():
    model = models.efficientnet_b0(pretrained=True)

    # Modify classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    return model.to(DEVICE)
```

### Custom Loss Function

```python
# training/train_rgb_model.py

import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Use in training:
criterion = FocalLoss(alpha=1, gamma=2)
```

---

## 🌐 API Extension

### Add New Endpoint

```python
# backend/server.py

@app.post("/batch_analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Analyze multiple images at once.

    Args:
        files: List of uploaded image files

    Returns:
        List of analysis results
    """
    results = []

    for file in files:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Analyze
        result = analyze_image(image, model=model_instance, return_base64=False)

        results.append({
            "filename": file.filename,
            "prediction": result["class_name"],
            "confidence": result["confidence"]
        })

    return {"results": results, "count": len(results)}
```

### Add Authentication

```python
# backend/server.py

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    token = credentials.credentials

    # Replace with your auth logic
    if token != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

    return token

# Protect endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)  # ← Add authentication
):
    # ... existing logic
```

---

## 🧪 Testing

### Unit Tests

Create `tests/test_metrics.py`:

```python
import pytest
import numpy as np
from solar_ai_system.training.metrics import compute_f1, compute_metrics

def test_compute_f1_perfect():
    """Test F1 with perfect predictions."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    f1 = compute_f1(y_true, y_pred, average='macro')
    assert f1 == 1.0

def test_compute_f1_zero():
    """Test F1 with completely wrong predictions."""
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]

    f1 = compute_f1(y_true, y_pred, average='macro')
    assert f1 == 0.0

def test_compute_metrics():
    """Test comprehensive metrics computation."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]  # 1 error

    class_names = ['Class0', 'Class1', 'Class2']
    metrics = compute_metrics(y_true, y_pred, class_names)

    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'per_class' in metrics
    assert metrics['accuracy'] > 0.8  # 5/6 correct
```

### API Tests

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from solar_ai_system.backend.server import app

client = TestClient(app)

def test_health_endpoint():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_classes_endpoint():
    """Test /classes endpoint."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 6

def test_analyze_endpoint_invalid_file():
    """Test /analyze with invalid file."""
    files = {'file': ('test.txt', b'invalid content', 'text/plain')}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
```

### Run Tests

```bash
pytest tests/ -v
```

---

## ✅ Best Practices

### Code Style

1. **Follow PEP 8**
2. **Use Black formatter:** `black solar_ai_system/`
3. **Type hints:** Always use type annotations
4. **Docstrings:** Google-style for all public functions

Example:

```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10
) -> Tuple[Dict[str, List[float]], float]:
    """
    Train the deep learning model.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs

    Returns:
        Tuple containing:
            - history: Dictionary with training metrics
            - best_acc: Best validation accuracy achieved

    Raises:
        ValueError: If model is not on the correct device
    """
    # Implementation here
    pass
```

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-backbone

# 2. Make changes
# ... code changes ...

# 3. Format code
black solar_ai_system/

# 4. Run tests
pytest tests/

# 5. Commit
git add .
git commit -m "Add ResNet50 backbone support"

# 6. Push
git push origin feature/new-backbone

# 7. Create Pull Request on GitHub
```

### Experiment Tracking

Always use the experiment logger:

```python
from training.experiment_logger import ExperimentLogger

# Create logger
logger = ExperimentLogger(experiment_name="resnet50_lr1e-4")

# Log hyperparameters
logger.log_hyperparameters(
    learning_rate=1e-4,
    batch_size=32,
    architecture="ResNet50"
)

# Log each epoch
for epoch in range(epochs):
    # ... training ...
    logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr, time)

# Log final metrics
logger.log_final_metrics(metrics)
logger.set_status("completed")
```

---

## 🚀 Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY solar_ai_system/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY solar_ai_system/ ./solar_ai_system/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "solar_ai_system/run_server.py"]
```

Build and run:

```bash
docker build -t solar-fault-ai .
docker run -p 8000:8000 solar-fault-ai
```

### Cloud Deployment (AWS)

1. **Package application:**
   ```bash
   zip -r solar-ai.zip solar_ai_system/ dataset/
   ```

2. **Upload to S3:**
   ```bash
   aws s3 cp solar-ai.zip s3://your-bucket/
   ```

3. **Deploy to EC2/ECS:**
   - Use Docker image
   - Or use Python virtual environment

4. **Set up Load Balancer:**
   - For production traffic handling

---

## 📝 Contributing Checklist

Before submitting a PR:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are added
- [ ] Tests are written and pass
- [ ] Code is formatted with Black
- [ ] No linting errors (flake8)
- [ ] Experiment logs are included
- [ ] README updated if needed
- [ ] CHANGELOG updated

---

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

**Happy Coding! 🚀**
