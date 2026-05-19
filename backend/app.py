"""
Malaria Detection - ONNX Runtime Version
Much smaller memory footprint for Render free tier
"""

import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (96, 96)
THRESHOLD = 0.5
session = None

def load_model():
    global session
    
    # Try to download model from GitHub or use local
    model_url = os.getenv("MODEL_URL", "")
    model_path = "model.onnx"
    
    if model_url and not os.path.exists(model_path):
        try:
            print(f"Downloading model from {model_url}")
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        except:
            pass
    
    if os.path.exists(model_path):
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ ONNX model loaded")
        return True
    else:
        print("⚠️ No model found - using dummy")
        return False

@app.on_event("startup")
async def startup():
    load_model()

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0).transpose(0, 3, 1, 2)  # ONNX wants NCHW

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Need an image file")
    
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 5 * 1024 * 1024:
            raise HTTPException(400, "Image too large")
        
        input_array = preprocess(image_bytes)
        
        if session:
            result = session.run(None, {'input': input_array})
            probability = float(result[0][0][0])
        else:
            # Fallback dummy prediction
            probability = 0.5
        
        is_infected = probability > THRESHOLD
        
        return {
            "has_malaria": is_infected,
            "confidence": round(probability if is_infected else 1 - probability, 3),
            "probability": round(probability, 3)
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": session is not None}

@app.get("/")
async def root():
    return {"service": "Malaria Detection API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)