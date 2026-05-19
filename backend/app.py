"""
Malaria Detection API - Optimized for Render.com Free Tier
Memory usage minimized for 512MB RAM limit
"""

import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import logging

# Minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/tiny_malaria_model.h5")
IMG_SIZE = (96, 96)
THRESHOLD = 0.5

# Global model
model = None

def load_model():
    """Load model with memory optimization"""
    global model
    
    try:
        # Import tensorflow only when needed
        import tensorflow as tf
        
        # Disable GPU and eager execution for memory saving
        tf.config.set_visible_devices([], 'GPU')
        tf.compat.v1.disable_eager_execution()
        
        # Set memory growth
        tf.config.experimental.set_memory_growth = True
        
        # Load model
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
            return True
        else:
            logger.error(f"Model not found at {MODEL_PATH}")
            return False
            
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return False

@app.on_event("startup")
async def startup():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict malaria from image"""
    if model is None:
        raise HTTPException(503, "Model loading. Try again in a moment.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        
        # Limit file size (Render free tier memory limit)
        if len(image_bytes) > 5 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 5MB)")
        
        # Process image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        import tensorflow as tf
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
        
        is_infected = probability > THRESHOLD
        confidence = probability if is_infected else 1 - probability
        
        return {
            "success": True,
            "has_malaria": is_infected,
            "confidence": round(confidence, 3),
            "probability": round(probability, 3)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Malaria Detection API",
        "status": "running",
        "endpoints": ["/health", "/predict"]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)