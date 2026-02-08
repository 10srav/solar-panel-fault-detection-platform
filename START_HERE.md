# 🚀 START HERE - Solar Panel Fault Detection AI

**BTech Final Year Project - Ready to Run!**

---

## 📁 **Your Clean Project**

```
C:\Users\BALU\OneDrive\Desktop\solar_panel\
│
├── backend/          ← Python API + AI (43MB model)
├── frontend/         ← React UI (beautiful interface)
├── dataset/          ← Training data (7,547 images)
└── unwanted/         ← Old files (can delete)
```

**Simple. Clean. Professional.** ✨

---

## ⚡ **RUN THE SYSTEM (2 Terminals)**

### **Terminal 1: Backend (4 commands)**

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_server.py
```

**Wait for:** `INFO: Uvicorn running on http://0.0.0.0:8000`

---

### **Terminal 2: Frontend (2 commands)**

```cmd
cd C:\Users\BALU\OneDrive\Desktop\solar_panel\frontend
npm install
npm start
```

**Browser opens at:** http://localhost:3000

---

## 🎯 **That's It!**

You now have:
- ✅ Backend API running on port 8000
- ✅ React UI running on port 3000
- ✅ AI model loaded (93.38% accuracy)
- ✅ Full-stack system ready!

---

## 🎨 **Using the React UI**

1. **Drag & drop** a solar panel image
2. **Click "Analyze Image"**
3. **See results:**
   - Fault type (Bird droppings, Dusty, Clean, etc.)
   - Confidence score with progress bar
   - Grad-CAM heatmap (where AI looked)
   - Risk level: 🟢 Low, 🟡 Medium, 🔴 High
   - Maintenance suggestion

---

## 📊 **For BTech Demo**

### **Sample Images:**

Use images from: `C:\Users\BALU\OneDrive\Desktop\solar_panel\dataset\PRoject\`

- `Bird_drop_generateds/` - Show bird dropping detection
- `Clean/` - Show clean panel detection
- `Dusty/` - Show dust accumulation detection
- `Electrical_damage_generated/` - Show electrical fault
- `Physcial_damage_generated/` - Show physical damage
- `Snow_covered_generated/` - Show snow coverage

### **Key Features to Demonstrate:**

1. **AI Accuracy** - 93.38% validation, 98.3% real-world test
2. **Real-time Processing** - <1 second per image
3. **Explainable AI** - Grad-CAM shows where model looks
4. **Risk Assessment** - Low/Medium/High categorization
5. **6 Fault Classes** - Comprehensive detection
6. **Modern UI** - Professional React interface

---

## 🔧 **If Something Goes Wrong**

### **Backend won't start?**

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### **Frontend won't start?**

```cmd
cd frontend
npm install
```

### **Port already in use?**

**Kill port 8000:**
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Kill port 3000:**
React will ask to use another port - type Y

---

## 📞 **Quick Reference**

| What | Where |
|------|-------|
| **Full Setup Guide** | `SETUP.md` |
| **Project Documentation** | `README.md` |
| **Quick Start** | `QUICKSTART.md` |
| **Cleanup Report** | `CLEANUP_SUMMARY.md` |

---

## 🌐 **Access Points**

| Service | URL |
|---------|-----|
| **React UI** | http://localhost:3000 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## 🎉 **Success Checklist**

- [ ] Terminal 1 shows: "Uvicorn running on http://0.0.0.0:8000"
- [ ] Terminal 2 shows: "webpack compiled successfully"
- [ ] Browser opens to http://localhost:3000
- [ ] You see the React UI with upload button
- [ ] Upload an image and see results

---

<p align="center">
  <strong>🌞 Your BTech Project is Ready! 🌞</strong>
</p>

<p align="center">
  <em>Clean • Professional • Production-Ready</em>
</p>

<p align="center">
  <strong>Open 2 terminals and run the commands above! 🚀</strong>
</p>
