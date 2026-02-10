# Solar Panel Fault Detection - Setup Guide

Everything you need to run the system. Follow the steps in order.

---

## What You Need (Install Once)

- **Python 3.10+** — Download from https://www.python.org/downloads/
- **Node.js 16+** — Download from https://nodejs.org/

Check if they are installed:

```
python --version
node --version
```

---

## First Time Setup (Do This Once)

### Step 1 — Backend Setup

Open a terminal (Command Prompt / PowerShell / Git Bash):

```
cd solar-panel-fault-detection-platform
cd backend
python -m venv venv
```

Activate the virtual environment:

- **Windows (Command Prompt):**
  ```
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```
  venv\Scripts\Activate.ps1
  ```
- **Mac / Linux:**
  ```
  source venv/bin/activate
  ```

You should see `(venv)` at the start of your terminal line.

Now install dependencies:

```
pip install -r requirements.txt
```

Wait for it to finish (takes 2-5 minutes).

### Step 2 — Frontend Setup

Open a **second terminal** (keep the first one open):

```
cd solar-panel-fault-detection-platform
cd frontend
npm install
```

Wait for it to finish (takes 2-3 minutes).

---

## Running the System (Every Time)

You need **2 terminals** running at the same time.

### Terminal 1 — Start Backend

```
cd solar-panel-fault-detection-platform/backend
venv\Scripts\activate
python -m api.server
```

You should see:

```
[API] Starting server at http://0.0.0.0:8000
[API] Model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open. Do not close it.

### Terminal 2 — Start Frontend

```
cd solar-panel-fault-detection-platform/frontend
npm start
```

You should see:

```
Compiled successfully!
```

Your browser will open automatically.

---

## Open the App

| What | URL |
|------|-----|
| **App (Main UI)** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## How to Use

The app has 3 tabs:

**Dashboard** — Shows prediction history and stats.

**Detection (RGB)** — Upload a regular solar panel photo.
- Detects: Bird droppings, Clean, Dusty, Electrical damage, Physical damage, Snow
- Shows: Fault type, confidence, Grad-CAM heatmap, risk level, maintenance advice

**Thermal Segmentation** — Upload a thermal/infrared image.
- Detects: Hotspot regions using U-Net segmentation
- Shows: Fault area %, severity, risk level, overlay mask, maintenance advice

---

## Quick Reference (3 Commands to Start)

After first-time setup, you only need these commands:

**Terminal 1:**
```
cd solar-panel-fault-detection-platform/backend
venv\Scripts\activate
python -m api.server
```

**Terminal 2:**
```
cd solar-panel-fault-detection-platform/frontend
npm start
```

Then open http://localhost:3000

---

## Stopping the System

Press `Ctrl + C` in both terminals.

---

## Troubleshooting

**"Port 8000 already in use"**
```
netstat -ano | findstr :8000
taskkill /PID <number> /F
```
Then start the backend again.

**"Module not found" error**
```
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend shows "Failed to fetch"**
Make sure Terminal 1 (backend) is still running. Check http://localhost:8000/health

**"No module named torch" or similar**
Make sure you activated the virtual environment first:
```
venv\Scripts\activate
```
You should see `(venv)` at the start of your terminal line.

---

## Models Included

Both AI models are included in this repo. No training needed.

| Model | File | Size | Purpose |
|-------|------|------|---------|
| RGB Classifier | `backend/models/rgb_fault_model.pth` | 43 MB | Fault classification (6 types) |
| Thermal U-Net | `backend/models/thermal_segmentation_unet_v2.pth` | 119 MB | Thermal hotspot segmentation |
