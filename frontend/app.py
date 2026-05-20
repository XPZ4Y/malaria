import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "model.h5"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DIRECT_MODEL_IMAGE_SIZE = (96, 96)

for directory in (UPLOAD_DIR, LOG_DIR):
    directory.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


def configure_logging():
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    app.logger.handlers.clear()
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(stream_handler)


configure_logging()
app.logger.info("Starting JWARA.AI server")

MODEL = load_model(MODEL_PATH)
app.logger.info("Loaded model from %s", MODEL_PATH)


def to_static_url(path):
    relative = Path(path).resolve().relative_to(BASE_DIR.resolve())
    return "/" + str(relative).replace("\\", "/")


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def normalize_patient_name(raw_name):
    cleaned = " ".join((raw_name or "").split())
    return cleaned


def predict_image_direct(image_path, model):
    with Image.open(image_path).convert("RGB") as image:
        image = image.resize(DIRECT_MODEL_IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array / 255.0

    batch = np.expand_dims(image_array, axis=0)
    probability = float(model.predict(batch, verbose=0)[0][0])
    infected = probability > 0.5
    confidence = probability if infected else 1.0 - probability

    diagnosis = "PARASITIZED (Malaria Detected)" if infected else "UNINFECTED (Healthy)"
    status = "positive" if infected else "negative"

    return {
        "diagnosis": diagnosis,
        "status": status,
        "confidence": round(confidence * 100.0, 2),
        "infection_rate": round(confidence * 100.0, 2),
        "model_probability": probability,
    }


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Image filename is empty"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    patient_name = normalize_patient_name(request.form.get("patient_name"))
    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400

    report_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    safe_name = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / f"{report_id}-{safe_name}"
    file.save(upload_path)

    app.logger.info("Processing report %s for %s", report_id, upload_path.name)

    try:
        analysis = predict_image_direct(str(upload_path), MODEL)
    except Exception as exc:
        app.logger.exception("Prediction failed for report %s", report_id)
        return jsonify({"error": str(exc)}), 500

    report = {
        "id": report_id,
        "patient_name": patient_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "filename": safe_name,
        "status": analysis["status"],
        "diagnosis": analysis["diagnosis"],
        "infection_rate": analysis["infection_rate"],
        "model_confidence": analysis["confidence"],
        "total_cells": None,
        "infected_cells": None,
        "healthy_cells": None,
        "quality_ok": None,
        "quality_message": "",
        "inference_mode": "direct_image",
        "model_probability": analysis["model_probability"],
    }

    app.logger.info(
        "Report %s complete: %s, infection_rate=%s",
        report_id,
        report["status"],
        report["infection_rate"],
    )
    return jsonify(report)


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
