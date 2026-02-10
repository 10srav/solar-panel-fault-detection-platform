"""
=============================================================================
 API SERVER — server.py
 FastAPI backend serving the inference pipeline + React frontend.
=============================================================================
"""

import os
import sys
import io
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Ensure backend/ is on the path for all imports
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BACKEND_DIR)

from config import (
    API_HOST, API_PORT, MAX_UPLOAD_SIZE_MB, VALID_EXTENSIONS,
    STATIC_DIR, CLASS_NAMES,
)
from inference.predictor import load_model, analyze_image
from inference.thermal_analyzer import analyze_thermal_image


# ============================================================================
# STARTUP: Pre-load model
# ============================================================================

model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at server startup."""
    global model_instance
    print("[API] Loading model at startup...")
    try:
        model_instance = load_model()
        print("[API] Model ready.")
    except FileNotFoundError as e:
        print(f"[API WARNING] {e}")
        print("[API] Server starting without model. Train first.")
    yield
    print("[API] Shutting down.")


# ============================================================================
# APP INIT
# ============================================================================

app = FastAPI(
    title="Solar Panel Fault Detection API",
    description="RGB image analysis with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend")
FRONTEND_REACT_DIR = os.path.join(BACKEND_DIR, "..", "frontend_react", "build")

# Try React build first, then plain frontend
if os.path.isdir(FRONTEND_REACT_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_REACT_DIR), name="frontend")
    SERVE_FRONTEND = FRONTEND_REACT_DIR
elif os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    SERVE_FRONTEND = FRONTEND_DIR
else:
    SERVE_FRONTEND = None


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Serve frontend or API info."""
    if SERVE_FRONTEND:
        index_path = os.path.join(SERVE_FRONTEND, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {
        "service": "Solar Panel Fault Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload image for analysis",
            "/health": "GET - Health check",
            "/classes": "GET - List fault classes",
            "/docs": "GET - Swagger API documentation",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_instance else "model_not_loaded",
        "model_loaded": model_instance is not None,
        "classes": CLASS_NAMES,
    }


@app.get("/classes")
async def get_classes():
    """Return available fault classes."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), input_type: str = Form("rgb")):
    """
    Main analysis endpoint (supports RGB and Thermal).

    Accepts:
        - file: image file upload (JPG, PNG, etc.)
        - input_type: "rgb" or "thermal" (default: "rgb")

    Returns: JSON with prediction/segmentation, severity, risk, suggestion
    """
    global model_instance

    # --- Validate model is loaded ---
    if model_instance is None:
        try:
            model_instance = load_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Train the model first.",
            )

    # --- Validate file type ---
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in VALID_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {ext}. Supported: {', '.join(VALID_EXTENSIONS)}",
            )

    # --- Read and validate file size ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {MAX_UPLOAD_SIZE_MB}MB",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Validate image can be opened ---
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot open file as image. File may be corrupted.",
        )

    # --- Run analysis (route based on input_type) ---
    try:
        if input_type.lower() == "thermal":
            # Thermal: U-Net segmentation only
            result = analyze_thermal_image(image, return_base64=True)

            response = {
                "success": True,
                "filename": file.filename,
                "input_type": "thermal",
                "analysis": {
                    "fault_area_percent": result["fault_area_percent"],
                    "fault_area_source": "unet_segmentation",
                    "severity_score": result["severity_score"],
                    "risk_level": result["risk_level"],
                    "maintenance_suggestion": result["maintenance_suggestion"],
                    "heat_difference": result.get("heat_difference", 0),
                    "image_mean_intensity": result.get("image_mean_intensity", 0),
                    "mask_mean_intensity": result.get("mask_mean_intensity", 0),
                },
                "alert": result["alert"],
                "segmentation_mask": result.get("segmentation_mask_base64", None),
                "segmentation_available": True,
            }
        else:
            # RGB: Classification + Grad-CAM
            result = analyze_image(
                image, model=model_instance, return_base64=True
            )

            response = {
                "success": True,
                "filename": file.filename,
                "input_type": "rgb",
                "prediction": {
                    "class_index": result["predicted_class"],
                    "class_name": result["class_name"],
                    "confidence": result["confidence"],
                },
                "class_probabilities": result.get("class_probabilities", {}),
                "analysis": {
                    "fault_area_percent": result["fault_area_percent"],
                    "fault_area_source": result.get("fault_area_source", "gradcam"),
                    "severity_score": result["severity_score"],
                    "risk_level": result["risk_level"],
                    "maintenance_suggestion": result["maintenance_suggestion"],
                },
                "alert": result["alert"],
                "gradcam_image": result.get("gradcam_base64", None),
                "segmentation_mask": result.get("segmentation_mask_base64", None),
                "segmentation_available": result.get("segmentation_available", False),
            }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )

    return JSONResponse(content=response)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"\n[API] Starting server at http://{API_HOST}:{API_PORT}")
    print(f"[API] Docs at http://localhost:{API_PORT}/docs\n")

    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
