# Solar Panel Fault Detection Platform

A production-ready, explainable AI system for detecting and classifying faults in solar panels using RGB imagery and thermal imaging. The platform combines deep learning classification, semantic segmentation, and explainability techniques to provide actionable insights for solar farm maintenance.

## Features

- **RGB Classification**: SparkNet-based CNN with Fire Modules for 6-class fault classification
  - Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-Covered
- **Thermal Segmentation**: U-Net model for pixel-level fault mask generation
- **Explainability**: Grad-CAM, Grad-CAM++, and Score-CAM for visual explanations
- **Severity Scoring**: Risk assessment combining fault area, temperature, and growth rate
- **REST API**: FastAPI backend with async endpoints and Prometheus metrics
- **Dashboard**: React + TypeScript frontend for monitoring and inference
- **Infrastructure**: Docker, CI/CD, PostgreSQL, MinIO (S3-compatible storage)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React)                           │
│  Dashboard │ Panel Details │ Inference │ Settings                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                               │
│  /infer/rgb │ /infer/thermal │ /infer/combined │ /panels │ /health     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  SparkNet │   │   U-Net   │   │  Grad-CAM │
            │  (RGB)    │   │ (Thermal) │   │  (XAI)    │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                          ┌─────────────────┐
                          │ Severity Scorer │
                          │ (Risk Level)    │
                          └─────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ PostgreSQL│   │   MinIO   │   │Prometheus │
            │   (DB)    │   │   (S3)    │   │ (Metrics) │
            └───────────┘   └───────────┘   └───────────┘
```

## Requirements

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- CUDA 11.8+ (optional, for GPU training)

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/solar-panel-fault-detection.git
cd solar-panel-fault-detection

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure the Application

```bash
# Copy default configuration
cp config/default.yaml config/local.yaml

# Edit config/local.yaml with your settings
# Key configurations:
#   - data.rgb_root: Path to RGB image dataset
#   - data.thermal_root: Path to thermal image dataset
#   - training.device: "cuda" or "cpu"
```

### 3. Prepare Data Folders

Organize your dataset in the following structure:

```
data/
├── rgb/
│   ├── train/
│   │   ├── Clean/
│   │   ├── Dusty/
│   │   ├── Bird-drop/
│   │   ├── Electrical-damage/
│   │   ├── Physical-damage/
│   │   └── Snow-Covered/
│   ├── val/
│   │   └── (same structure as train)
│   └── test/
│       └── (same structure as train)
├── thermal/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── models/
    ├── sparknet_best.pth
    └── unet_best.pth
```

**RGB Images**: JPEG/PNG images of solar panels (recommended: 227x227)
**Thermal Images**: Single-channel thermal images with corresponding binary masks

### 4. Train Models

#### Train SparkNet (RGB Classification)

```bash
python scripts/train_sparknet.py \
    --config config/local.yaml \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir outputs/sparknet
```

**Training options:**
- `--resume`: Path to checkpoint to resume training
- `--device`: Override device (cuda/cpu)
- `--wandb`: Enable Weights & Biases logging
- `--augment`: Enable data augmentation

#### Train U-Net (Thermal Segmentation)

```bash
python scripts/train_unet.py \
    --config config/local.yaml \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --output-dir outputs/unet
```

**Training options:**
- `--loss`: Loss function (dice, bce, dice_bce)
- `--features`: Encoder feature dimensions (default: 64,128,256,512)
- `--attention`: Enable attention gates

### 5. Run Evaluation and Ablation

#### Evaluate SparkNet

```bash
python scripts/evaluate.py sparknet \
    --checkpoint outputs/sparknet/best_model.pth \
    --data-dir data/rgb/test \
    --output-dir outputs/evaluation/sparknet
```

Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- ROC curves for each class
- Sample predictions with Grad-CAM visualizations

#### Evaluate U-Net

```bash
python scripts/evaluate.py unet \
    --checkpoint outputs/unet/best_model.pth \
    --data-dir data/thermal/test \
    --output-dir outputs/evaluation/unet
```

Generates:
- Segmentation metrics (IoU, Dice, pixel accuracy)
- Sample segmentation visualizations
- Error analysis by fault type

#### Run Ablation Studies

```bash
# Ablation on Fire Module configurations
python scripts/ablation.py fire-modules \
    --config config/local.yaml \
    --output-dir outputs/ablation

# Ablation on severity scoring weights
python scripts/ablation.py severity-weights \
    --config config/local.yaml \
    --output-dir outputs/ablation
```

### 6. Start API Server

#### Development Mode

```bash
python scripts/serve.py \
    --config config/local.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

#### Production Mode

```bash
python scripts/serve.py \
    --config config/production.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

API will be available at:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Metrics: http://localhost:8000/metrics

### 7. Run Frontend

```bash
cd src/frontend

# Install dependencies
npm install

# Development mode
npm run dev

# Production build
npm run build
npm run preview
```

Frontend will be available at http://localhost:5173

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

Services:
- **api**: FastAPI backend (port 8000)
- **frontend**: React dashboard (port 80)
- **db**: PostgreSQL database (port 5432)
- **minio**: S3-compatible storage (port 9000, console: 9001)
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Dashboards (port 3000)

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/solar_detection

# S3/MinIO
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=solar-images

# Model paths
SPARKNET_MODEL_PATH=/app/models/sparknet_best.pth
UNET_MODEL_PATH=/app/models/unet_best.pth

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns service health status and model loading state.

#### RGB Inference
```
POST /infer/rgb
Content-Type: multipart/form-data

Parameters:
- image: RGB image file
- generate_gradcam: bool (optional, default: true)
```

Response:
```json
{
  "class_name": "Dusty",
  "class_id": 1,
  "confidence": 0.94,
  "probabilities": {
    "Clean": 0.02,
    "Dusty": 0.94,
    "Bird-drop": 0.01,
    ...
  },
  "gradcam_image": "base64...",
  "inference_time_ms": 45.2
}
```

#### Thermal Inference
```
POST /infer/thermal
Content-Type: multipart/form-data

Parameters:
- image: Thermal image file
- threshold: float (optional, default: 0.5)
```

Response:
```json
{
  "mask": "base64...",
  "fault_area_ratio": 0.15,
  "overlay_image": "base64...",
  "inference_time_ms": 32.1
}
```

#### Combined Inference
```
POST /infer/combined
Content-Type: multipart/form-data

Parameters:
- rgb_image: RGB image file
- thermal_image: Thermal image file
```

Response:
```json
{
  "classification": { ... },
  "segmentation": { ... },
  "severity": {
    "fault_area_ratio": 0.15,
    "temperature_score": 0.72,
    "growth_rate": 0.05,
    "severity_score": 0.48,
    "risk_level": "Medium",
    "alert_triggered": false
  }
}
```

#### Panel Management
```
GET /panels                    # List all panels
POST /panels                   # Create panel
GET /panels/{panel_id}         # Get panel details
PUT /panels/{panel_id}         # Update panel
DELETE /panels/{panel_id}      # Delete panel
GET /panels/{panel_id}/history # Get fault history
```

## Configuration Reference

```yaml
# config/default.yaml

data:
  rgb_root: "data/rgb"
  thermal_root: "data/thermal"
  image_size: 227
  thermal_size: 256
  num_workers: 4
  pin_memory: true

model:
  sparknet:
    num_classes: 6
    input_channels: 3
    dropout_rate: 0.5
  unet:
    in_channels: 1
    out_channels: 1
    features: [64, 128, 256, 512]
    attention: true

training:
  device: "cuda"
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  mixed_precision: true

severity:
  weights:
    area: 0.4
    temperature: 0.4
    growth: 0.2
  thresholds:
    low: 0.3
    high: 0.7
  alert_threshold: 0.8

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["*"]

database:
  url: "postgresql://user:pass@localhost:5432/solar_detection"
  pool_size: 5
  max_overflow: 10

storage:
  type: "s3"
  endpoint: "http://localhost:9000"
  bucket: "solar-images"
  access_key: "minioadmin"
  secret_key: "minioadmin"
```

## Model Architecture

### SparkNet (RGB Classification)

Based on the SparkNet architecture from "SparkNet - A Solar Panel Fault Detection Deep Learning Model" (IEEE Access 2025).

Key components:
- **Fire Modules**: Squeeze (1x1 conv) + Expand (1x1 and 3x3 conv) for efficient feature extraction
- **4-Branch Hierarchical Structure**: Multi-scale feature aggregation
- **Global Average Pooling**: Reduces overfitting
- **Dropout**: 50% dropout for regularization

Architecture:
```
Input (3, 227, 227)
    ↓
Conv1 (96 filters, 7x7, stride 2)
    ↓
MaxPool (3x3, stride 2)
    ↓
Fire2 → Fire3 → Fire4 → Fire5
    ↓
MaxPool (3x3, stride 2)
    ↓
Fire6 → Fire7 → Fire8 → Fire9
    ↓
Dropout (0.5)
    ↓
Conv10 (num_classes filters, 1x1)
    ↓
Global Average Pooling
    ↓
Softmax
```

### U-Net (Thermal Segmentation)

Standard U-Net architecture with attention gates:

```
Encoder                    Decoder
────────                   ────────
Conv Block (64)      ←──── Up Block (64)
    ↓                          ↑
Conv Block (128)     ←──── Up Block (128)
    ↓                          ↑
Conv Block (256)     ←──── Up Block (256)
    ↓                          ↑
Conv Block (512)     ←──── Up Block (512)
    ↓                          ↑
         Bottleneck (1024)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run integration tests
pytest tests/integration/
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

1. **Lint**: Ruff linting and formatting checks
2. **Type Check**: MyPy static type analysis
3. **Unit Tests**: Pytest with coverage
4. **Build**: Docker image build
5. **Push**: Push to container registry (on main branch)

## Monitoring

### Prometheus Metrics

Available at `/metrics`:
- `inference_requests_total`: Total inference requests by endpoint
- `inference_duration_seconds`: Inference latency histogram
- `model_predictions_total`: Predictions by class
- `active_panels_total`: Number of monitored panels

### Grafana Dashboards

Pre-configured dashboards:
- **Inference Performance**: Latency, throughput, error rates
- **Model Metrics**: Prediction distribution, confidence scores
- **System Health**: CPU, memory, GPU utilization

## License

MIT License - see LICENSE file for details.

## Citation

If you use this platform in your research, please cite:

```bibtex
@article{sparknet2025,
  title={SparkNet - A Solar Panel Fault Detection Deep Learning Model},
  journal={IEEE Access},
  year={2025},
  volume={13},
  pages={1-15},
  doi={10.1109/ACCESS.2025.XXXXXXX}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas
