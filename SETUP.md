# 🚀 Solar Panel Fault Detection AI - Windows Setup Guide

**BTech Final Year Project - Complete Setup in 5 Minutes**

---

## 📁 Project Structure

```
C:\Users\BALU\OneDrive\Desktop\solar_panel\
├── backend/          ← Python API + AI Model (FastAPI)
├── frontend/         ← React UI Dashboard
├── dataset/          ← Training data (7,547 images)
└── unwanted/         ← Old files (IGNORE THIS)
```

**Model Already Trained:** `backend/models/rgb_fault_model.pth` (43MB, 90.9% accuracy) ✅

---

## ⚡ Quick Start - First Time Setup

### **Step 1: Install Prerequisites** (One-time only)

#### ✅ Check if Python is installed:
```cmd
python --version
```
Should show: `Python 3.10` or higher

If not installed: Download from https://www.python.org/downloads/

#### ✅ Check if Node.js is installed:
```cmd
node --version
npm --version
```
Should show: `v16.0.0` or higher

If not installed: Download from https://nodejs.org/

---

### **Step 2: Setup Backend** (One-time only)

Open **Command Prompt (Terminal 1)** and run:

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\backend
pip install -r requirements.txt
```

⏱️ Takes ~2 minutes. You should see packages installing (torch, fastapi, etc.)

---

### **Step 3: Setup Frontend** (One-time only)

Open **ANOTHER Command Prompt (Terminal 2)** and run:

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\frontend
npm install
```

⏱️ Takes ~3 minutes. You should see "267 packages installed"

---

## 🚀 Running the System (Every Time)

### **Terminal 1: Start Backend API**

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\backend
python run_server.py
```

✅ **SUCCESS when you see:**
```
  Solar Panel Fault Detection API
  Dashboard: http://localhost:8000
  API Docs:  http://localhost:8000/docs

[MODEL] Loaded: ...\rgb_fault_model.pth
[MODEL] Best val acc: 90.93% (epoch 2) | Device: cuda
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**🔗 Backend URLs:**
- **API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health
- **API Documentation:** http://localhost:8000/docs

⚠️ **Keep this terminal running!** Don't close it.

---

### **Terminal 2: Start Frontend UI**

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\frontend
npm start
```

✅ **SUCCESS when you see:**
```
Compiled successfully!

webpack compiled with 1 warning
```

Your browser will **auto-open** to: **http://localhost:3000**

**🔗 Frontend URL:**
- **React Dashboard:** http://localhost:3000

⚠️ **Keep this terminal running!** Don't close it.

---

## 🎯 Using the System

### Upload & Analyze Image:

1. **Open:** http://localhost:3000
2. **Drag & drop** a solar panel image (or click to browse)
3. **Click "Analyze Panel"**
4. **View Results:**
   - ✅ Fault Type Detected
   - ✅ Confidence Score (%)
   - ✅ Grad-CAM Heatmap (AI's focus areas)
   - ✅ Fault Area Percentage
   - ✅ Severity Score
   - ✅ Risk Level (Low/Medium/High)
   - ✅ Maintenance Recommendation

### Test with Sample Images:

```
C:\Users\BALU\OneDrive\Desktop\solar_panel\dataset\PRoject\
├── Bird_drop_generateds/     ← Test bird dropping detection
├── Clean/                    ← Test clean panel
├── Dusty/                    ← Test dust detection
├── Electrical_damage_generated/  ← Test electrical fault
├── Physcial_damage_generated/    ← Test physical damage
└── Snow_covered_generated/       ← Test snow detection
```

---

## 🛑 Stop the System

Press **Ctrl+C** in **both terminals** to stop.

---

## 🔄 Quick Restart (After First Setup)

You only need to run these 2 commands (no setup needed):

**Terminal 1:**
```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\backend
python run_server.py
```

**Terminal 2:**
```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\frontend
npm start
```

---

## 🔧 Troubleshooting

### ❌ "Port 8000 already in use"

Kill the process:
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### ❌ "Port 3000 already in use"

React will ask: **"Would you like to run on another port? (Y/n)"**

Type `Y` and press Enter → it will use port 3001

### ❌ "Module not found" error

**Backend:**
```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\backend
pip install -r requirements.txt
```

**Frontend:**
```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\frontend
npm install
```

### ❌ Frontend shows "Failed to fetch"

**Check:**
1. Backend is running (Terminal 1 should say "Uvicorn running")
2. Visit: http://localhost:8000/health
3. Should show: `{"status":"healthy","model_loaded":true}`

If not, restart Terminal 1.

---

## 📊 System Specifications

| Feature | Details |
|---------|---------|
| **Model** | ResNet18 (Transfer Learning) |
| **Accuracy** | 90.93% validation, 98.3% test |
| **Classes** | 6 fault types |
| **Dataset** | 7,547 images |
| **Inference Time** | <1 second per image |
| **Backend** | FastAPI + PyTorch + CUDA |
| **Frontend** | React 18 |
| **Device** | CUDA (GPU) |

---

## 🌐 All Access URLs

| Service | URL | Description |
|---------|-----|-------------|
| **React Dashboard** | http://localhost:3000 | Main UI - Upload & analyze images |
| **Backend API** | http://localhost:8000 | FastAPI server |
| **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/health | Check if model is loaded |
| **Classes List** | http://localhost:8000/classes | List all 6 fault types |

---

## ✅ Final Checklist

- [x] Model trained (90.93% accuracy) ✅
- [ ] Python 3.10+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed (`pip install`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Terminal 1 running backend (`python run_server.py`)
- [ ] Terminal 2 running frontend (`npm start`)
- [ ] Browser shows React UI at http://localhost:3000
- [ ] Successfully analyzed a test image

---

## 🎓 For BTech Project Demo

### Demo Flow:

1. **Show System Architecture** (backend + frontend + model)
2. **Upload Clean Panel** → Show "No action needed" result
3. **Upload Dusty Panel** → Show "Surface cleaning" recommendation
4. **Upload Electrical Damage** → Show "CRITICAL FAULT" alert
5. **Explain Grad-CAM Heatmap** → Visual explainability
6. **Show API Documentation** → http://localhost:8000/docs
7. **Highlight 90.93% Accuracy** + Real-time inference

---

## 🎉 Success Indicators

✅ **Terminal 1 (Backend):**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Terminal 2 (Frontend):**
```
webpack compiled with 1 warning
```

✅ **Browser:**
- Clean React UI with upload button
- Dark theme with modern design
- "Solar Panel Fault Detection" header

---

<p align="center">
  <strong>🌞 System Ready for Demo! 🌞</strong>
</p>

<p align="center">
  <b>Backend:</b> http://localhost:8000 |
  <b>Frontend:</b> http://localhost:3000
</p>

<p align="center">
  <em>90.93% Accuracy | 6 Fault Types | Real-time Analysis</em>
</p>
