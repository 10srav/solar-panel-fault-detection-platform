# ⚡ Quick Start - 5 Minutes to Running System

**Get the Solar Panel Fault Detection AI running in 5 minutes!**

---

## 🎯 What You'll Do

1. Install dependencies (2 min)
2. Start the API server (30 sec)
3. Upload an image and get predictions! (30 sec)

---

## 📋 Prerequisites

- ✅ Python 3.10 or higher installed
- ✅ Windows, Linux, or macOS
- ✅ (Optional) CUDA-capable GPU for faster training

---

## 🚀 Quick Setup (3-4 Commands)

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
cd solar_ai_system && pip install -r requirements.txt
python run_server.py
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
cd solar_ai_system && pip install -r requirements.txt
python run_server.py
```

**📖 For detailed setup with frontend, see [SETUP.md](SETUP.md)**

---

## ✅ Verify It's Working

You should see:

```
Solar Panel Fault Detection API
Dashboard: http://localhost:8000
API Docs:  http://localhost:8000/docs

INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 🌐 Use the System

### Method 1: Web Dashboard (Easiest)

1. Open browser: **http://localhost:8000**
2. Upload a solar panel image
3. View prediction + Grad-CAM + risk level!

### Method 2: API (for Developers)

```python
import requests

url = "http://localhost:8000/analyze"
files = {'file': open('panel_image.jpg', 'rb')}
response = requests.post(url, files=files)

result = response.json()
print(f"Fault: {result['prediction']['class_name']}")
print(f"Confidence: {result['prediction']['confidence']:.1f}%")
```

### Method 3: cURL (Command Line)

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@panel_image.jpg"
```

---

## 📊 Example Output

```json
{
  "success": true,
  "prediction": {
    "class_name": "Dusty",
    "confidence": 95.3
  },
  "analysis": {
    "risk_level": "Medium",
    "severity_score": 35.8,
    "maintenance_suggestion": "Surface cleaning — remove dust"
  }
}
```

---

## 🔧 Troubleshooting

### Issue: Port 8000 already in use

**Solution:** Change port in `solar_ai_system/config.py`:

```python
API_PORT = 8001  # or any other port
```

### Issue: ModuleNotFoundError

**Solution:** Make sure you installed dependencies:

```bash
cd solar_ai_system
pip install -r requirements.txt
```

### Issue: Model not found

**Solution:** The pre-trained model should be at `solar_ai_system/models/rgb_fault_model.pth`. If missing, train the model:

```bash
python solar_ai_system/run_training.py
```

---

## 📚 Next Steps

### For Users
- ✅ **Try different images** - Test various solar panel faults
- ✅ **Check API docs** - Visit http://localhost:8000/docs
- ✅ **View results** - See plots in `solar_ai_system/plots/`

### For Developers
- ✅ **Read README.md** - Full system documentation
- ✅ **Check DEVELOPER_GUIDE.md** - Technical details
- ✅ **Explore notebooks** - `solar_ai_system/notebooks/`
- ✅ **Customize config** - Edit `solar_ai_system/config.py`

### For Researchers
- ✅ **Retrain model** - `python run_training.py`
- ✅ **View metrics** - Check `solar_ai_system/training/logs/`
- ✅ **Analyze results** - Open Jupyter notebooks

---

## 🎯 Training the Model (Optional)

If you want to retrain with your own data:

```bash
# 1. Organize your dataset
dataset/
└── PRoject/
    ├── Class1/
    ├── Class2/
    └── ...

# 2. Update class names in config.py (if needed)

# 3. Run training
python solar_ai_system/run_training.py

# Wait ~35 minutes (RTX 3050) or ~2 hours (CPU)
```

---

## ✨ Key Features to Try

1. **Upload Image** → Get instant fault detection
2. **View Grad-CAM** → See where the model looked
3. **Check Risk Level** → Understand maintenance priority
4. **Read Suggestion** → Get actionable recommendations

---

## 📞 Need Help?

- **Documentation:** Read [README.md](README.md)
- **API Reference:** http://localhost:8000/docs
- **Issues:** Check troubleshooting section above
- **Contact:** See README.md for support info

---

## 🎉 You're All Set!

The system is now running and ready to detect solar panel faults!

**What's Next?**
- Upload your first image at http://localhost:8000
- Explore the API documentation
- Try the Jupyter notebook tutorial
- Read the comprehensive README.md

---

<p align="center">
  <strong>Happy Fault Detection! 🌞</strong>
</p>
