"""
Malaria Detection API - Render Deployment
Compatible with TensorFlow 2.13.0
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Load TensorFlow with version check
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/tiny_malaria_model.h5')
IMG_SIZE = (96, 96)
CONFIDENCE_THRESHOLD = 0.5

# Global model variable
model = None

def load_model():
    """Load the Keras model with fallback options"""
    global model
    
    # Try primary path
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Failed to load from {MODEL_PATH}: {e}")
    
    # Try alternative paths
    alt_paths = [
        'tiny_malaria_model.h5',
        'best_tiny_model.h5',
        'models/best_tiny_model.h5',
        '../tiny_malaria_model.h5'
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Model loaded from {path}")
                return True
            except Exception as e:
                print(f"❌ Failed to load from {path}: {e}")
    
    return False

def preprocess_image(image_bytes):
    """Convert uploaded image to model-ready format"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    
    except Exception as e:
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'unavailable'
        }), 503
    
    # Get image from request
    if 'image' not in request.files and 'image_base64' not in request.json:
        return jsonify({
            'error': 'No image provided. Send as multipart/form-data with key "image" or JSON with "image_base64"'
        }), 400
    
    try:
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            image_bytes = file.read()
        
        # Handle base64 JSON
        elif 'image_base64' in request.json:
            base64_str = request.json['image_base64']
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
        
        else:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Preprocess image
        processed_image, error = preprocess_image(image_bytes)
        if error:
            return jsonify({'error': f'Image preprocessing failed: {error}'}), 400
        
        # Run inference
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        # Determine result
        is_infected = probability > CONFIDENCE_THRESHOLD
        confidence = probability if is_infected else 1 - probability
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'has_malaria': is_infected,
                'probability_infected': round(probability, 4),
                'probability_healthy': round(1 - probability, 4),
                'confidence': round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2)
            },
            'model_info': {
                'input_size': IMG_SIZE,
                'threshold': CONFIDENCE_THRESHOLD
            }
        }
        
        # Add diagnosis message
        if is_infected:
            response['prediction']['diagnosis'] = 'MALARIA DETECTED'
            response['prediction']['diagnosis_code'] = 'POSITIVE'
        else:
            response['prediction']['diagnosis'] = 'NO MALARIA'
            response['prediction']['diagnosis_code'] = 'NEGATIVE'
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple images"""
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({'error': 'No images in batch'}), 400
    
    results = []
    for idx, file in enumerate(files):
        try:
            image_bytes = file.read()
            processed_image, error = preprocess_image(image_bytes)
            
            if error:
                results.append({
                    'index': idx,
                    'filename': file.filename,
                    'error': error
                })
                continue
            
            prediction = model.predict(processed_image, verbose=0)
            probability = float(prediction[0][0])
            is_infected = probability > CONFIDENCE_THRESHOLD
            
            results.append({
                'index': idx,
                'filename': file.filename,
                'has_malaria': is_infected,
                'probability': round(probability, 4),
                'confidence': round(probability if is_infected else 1 - probability, 4)
            })
        
        except Exception as e:
            results.append({
                'index': idx,
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results
    }), 200

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_loaded': True,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': model.count_params(),
        'tensorflow_version': tf.__version__
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Main entry point
if __name__ == '__main__':
    print("=" * 50)
    print("🦟 Malaria Detection API")
    print("=" * 50)
    
    # Load model before starting server
    if load_model():
        port = int(os.environ.get('PORT', 5000))
        print(f"\n🚀 Starting server on port {port}")
        print(f"📍 Health check: http://localhost:{port}/health")
        print(f"📍 Model info: http://localhost:{port}/info")
        print(f"📍 Predict endpoint: POST http://localhost:{port}/predict")
        print("\n✅ API ready for requests\n")
        
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("\n❌ Failed to load model. Please check MODEL_PATH")
        exit(1)