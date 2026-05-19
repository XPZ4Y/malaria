"""
Malaria Detection API - FastAPI Backend
---------------------------------------
Hosts the .h5 model for real-time inference
"""

import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Malaria Detection API",
    description="AI-powered malaria detection from blood cell images",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "models/tiny_malaria_model.h5"
IMG_SIZE = (96, 96)
CLASS_THRESHOLD = 0.5

# Global model variable
model = None

def load_model():
    """Load the trained Keras model"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"✅ Image preprocessed: shape={img_array.shape}")
        return img_array
    
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def predict(image_array: np.ndarray) -> Dict[str, Any]:
    """
    Run model inference
    
    Args:
        image_array: Preprocessed image array
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Run prediction
        prediction = model.predict(image_array, verbose=0)
        probability = float(prediction[0][0])
        
        # Determine class
        is_infected = probability > CLASS_THRESHOLD
        confidence = probability if is_infected else 1 - probability
        class_label = "Parasitized" if is_infected else "Uninfected"
        
        return {
            "success": True,
            "prediction": {
                "has_malaria": is_infected,
                "class": class_label,
                "probability_infected": round(probability, 4),
                "probability_healthy": round(1 - probability, 4),
                "confidence": round(confidence, 4),
                "threshold_used": CLASS_THRESHOLD
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Malaria Detection API...")
    if not load_model():
        logger.warning("⚠️ Model not loaded. API will not function properly.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Malaria Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_base64": "/predict/base64 (POST)",
            "info": "/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "input_size": IMG_SIZE,
        "threshold": CLASS_THRESHOLD,
        "total_parameters": model.count_params()
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict malaria from uploaded image file
    
    Args:
        file: Image file (jpg, png, jpeg)
    
    Returns:
        Prediction results
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Expected image, got {file.content_type}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size 10MB")
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Run prediction
        result = predict(img_array)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/base64")
async def predict_base64(data: Dict[str, str]):
    """
    Predict malaria from base64 encoded image
    
    Args:
        data: JSON with 'image' field containing base64 encoded image
    
    Returns:
        Prediction results
    """
    import base64
    
    try:
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field")
        
        # Decode base64
        image_bytes = base64.b64decode(data['image'])
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Run prediction
        result = predict(img_array)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict malaria for multiple images
    
    Args:
        files: List of image files
    
    Returns:
        Batch prediction results
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": f"Invalid file type: {file.content_type}"
                })
                continue
            
            # Read and process image
            image_bytes = await file.read()
            if len(image_bytes) > 10 * 1024 * 1024:
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": "File too large (max 10MB)"
                })
                continue
            
            img_array = preprocess_image(image_bytes)
            result = predict(img_array)
            
            results.append({
                "index": idx,
                "filename": file.filename,
                **result
            })
        
        except Exception as e:
            errors.append({
                "index": idx,
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_processed": len(results),
        "total_errors": len(errors),
        "results": results,
        "errors": errors
    })

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )