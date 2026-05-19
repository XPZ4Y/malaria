"""
Malaria Detection API - Render.com Deployment
Run with: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
from typing import Dict, Any, List
import uvicorn

# ============================================
# CONFIGURATION
# ============================================

IMG_SIZE = (96, 96)
# Get model path - works both locally and on Render
MODEL_PATH = os.environ.get("MODEL_PATH", "models/tiny_malaria_model.h5")

# ============================================
# MODEL LOADER
# ============================================

class MalariaDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load Keras (.h5) model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            
            print(f"✓ Model loaded: {self.model_path}")
            # print(self.model.summary())
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image to match training format"""
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(IMG_SIZE)
        img_array = np.array(image)
        
        # Standardize/Normalize as per training (usually /255.0)
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run inference on image"""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            # Preprocess
            img_array = self.preprocess_image(image_bytes)
            
            # Prepare input (add batch dimension)
            input_data = np.expand_dims(img_array, axis=0)
            
            # Run inference
            output = self.model.predict(input_data)
            
            probability = float(output[0][0])
            has_malaria = probability > 0.5
            confidence = probability if has_malaria else 1 - probability
            
            return {
                "success": True,
                "has_malaria": has_malaria,
                "probability": round(probability, 4),
                "confidence": round(confidence, 4),
                "probability_percent": round(probability * 100, 2),
                "confidence_percent": round(confidence * 100, 2),
                "diagnosis": "MALARIA DETECTED" if has_malaria else "NO MALARIA"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# ============================================
# INITIALIZE API
# ============================================

app = FastAPI(
    title="Malaria Detection API",
    description="AI-powered malaria detection from blood cell images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = MalariaDetector(MODEL_PATH)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    print("🦟 Loading Malaria Detection Model...")
    detector.load_model()
    print("✅ API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 Shutting down...")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Malaria Detection API",
        "status": "running",
        "model_loaded": detector.is_loaded,
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector.is_loaded,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict malaria from uploaded image
    
    Example:
        curl -X POST -F "file=@image.jpg" https://your-app.onrender.com/predict
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and predict
    image_bytes = await file.read()
    result = detector.predict(image_bytes)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return JSONResponse(content=result)

@app.post("/predict/base64")
async def predict_base64(data: dict):
    """
    Predict malaria from base64 encoded image
    
    Request: {"image": "base64_string"}
    """
    try:
        image_b64 = data.get("image", "")
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(image_b64)
        result = detector.predict(image_bytes)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {str(e)}")

# ============================================
# RUN SERVER (for local development)
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )