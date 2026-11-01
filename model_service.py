"""TensorFlow.js model conversion, loading, image validation, and inference."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError


MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_JSON = MODEL_DIR / "model.json"
METADATA_JSON = MODEL_DIR / "metadata.json"
CONVERTED_MODEL = MODEL_DIR / "crop_disease_model.h5"
MIN_CONFIDENCE = float(os.getenv("CROP_MODEL_MIN_CONFIDENCE", "0.70"))
MIN_IMAGE_SIDE = int(os.getenv("CROP_MODEL_MIN_IMAGE_SIDE", "32"))
MAX_UPLOAD_BYTES = int(os.getenv("CROP_MODEL_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Exact mapping of the existing Teachable Machine labels to crop/database names.
CLASS_MAPPING = {
    "Healthy(Potato)": ("Potato", "Healthy"), "Late blight1": ("Potato", "Late blight"),
    "Early Blight1": ("Potato", "Early blight"), "Black scurf and stem canker ": ("Potato", "Black scurf and stem canker"),
    "Common scab": ("Potato", "Common scab"), "HEALTHY(RICE)": ("Rice", "Healthy"),
    "Bacterial leaf blight": ("Rice", "Bacterial leaf blight"), "brown spot": ("Rice", "Brown spot"),
    "Sheath blight": ("Rice", "Sheath blight"), "False smut": ("Rice", "False smut"), "sheath rot": ("Rice", "Sheath rot"),
    "healthy(sugarcane)": ("Sugarcane", "Healthy"), "Red rot": ("Sugarcane", "Red rot"),
    "Sugarcane smut": ("Sugarcane", "Sugarcane smut"), "Grassy shoot disease": ("Sugarcane", "Grassy shoot disease"),
    "Sugarcane mosaic": ("Sugarcane", "Sugarcane mosaic"), "Tomato": ("Tomato", "Healthy"),
    "Late blight": ("Tomato", "Late blight"), "Early blight": ("Tomato", "Early blight"),
    "septoria leaf spot": ("Tomato", "Septoria leaf spot"), "Healthy(wheat)": ("Wheat", "Healthy"),
    "leaft rust(brown rust": ("Wheat", "Leaf rust (brown rust)"), "steam rust(black rust)": ("Wheat", "Stem rust (black rust)"),
    "Yellow rust": ("Wheat", "Yellow rust (stripe rust)"), "loose smut": ("Wheat", "Loose smut"), "flag smut": ("Wheat", "Flag smut"),
    "Healthy(onion)": ("Onion", "Healthy"), "Purple blotch": ("Onion", "Purple blotch"),
    "Downy mildew": ("Onion", "Downy mildew"), "stemphylium leaf blight": ("Onion", "Stemphylium leaf blight"),
    "Bulb Blight-D": ("Onion", "Bulb Blight-D"), "Rust": ("Onion", "Rust"), "Healthy(maize)": ("Maize", "Healthy"),
    "Maydis leaf blight": ("Maize", "Maydis leaf blight"), "Northern corn leaf blight": ("Maize", "Northern corn leaf blight"),
    "Gray leaf spot": ("Maize", "Gray leaf spot"), "Common rust": ("Maize", "Common rust"), "Insect damage": ("Maize", "Insect damage"),
    "Healthy(Banana)": ("Banana", "Healthy"), "Black sigatoka": ("Banana", "Black Sigatoka"),
    "Yellow sigatoka": ("Banana", "Yellow Sigatoka"), "Banana bunchy top": ("Banana", "Banana bunchy top"),
    "Banana Bract mosaic": ("Banana", "Banana bract mosaic"), "Fusarium wilt(Panama Disease)": ("Banana", "Fusarium wilt (Panama disease)"),
    "Moko Bacterial wilt": ("Banana", "Moko bacterial wilt"),
}


class PredictionError(Exception):
    def __init__(self, error_type: str, message: str, status_code: int = 422) -> None:
        self.error_type, self.message, self.status_code = error_type, message, status_code
        super().__init__(message)


class CropDiseaseModel:
    def __init__(self) -> None:
        self.model = None
        self.labels: list[str] = []

    def load(self) -> None:
        if self.model is not None:
            return
        if not MODEL_JSON.exists() or not METADATA_JSON.exists() or not (MODEL_DIR / "weights.bin").exists():
            raise RuntimeError("Required Teachable Machine model files are missing")
        if not CONVERTED_MODEL.exists():
            command = [sys.executable, "-m", "tensorflowjs.converters.converter", "--input_format=tfjs_layers_model", "--output_format=keras", str(MODEL_JSON), str(CONVERTED_MODEL)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Could not convert the supplied TensorFlow.js model: {completed.stderr[-800:]}")
        import tensorflow as tf  # Deferred so installation errors are clear at startup.
        with METADATA_JSON.open(encoding="utf-8") as source:
            self.labels = json.load(source).get("labels", [])
        self.model = tf.keras.models.load_model(CONVERTED_MODEL, compile=False)

    def predict(self, content: bytes) -> dict:
        if not content:
            raise PredictionError("missing_image", "Please choose an image before scanning.", 400)
        if len(content) > MAX_UPLOAD_BYTES:
            raise PredictionError("image_too_large", "Please upload an image smaller than 10 MB.", 413)
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("invalid_image", "Please upload a readable image file.", 400) from exc
        if min(image.size) < MIN_IMAGE_SIDE:
            raise PredictionError("invalid_image", "Please upload a larger, clear crop or plant image.")
        if max(sum(channel) / len(channel) for channel in ImageStat.Stat(image.resize((32, 32))).var) < 5:
            raise PredictionError("invalid_image", "Please upload a clear image of a supported crop or plant.")
        array = np.asarray(image.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.float32)
        array = (array / 127.5) - 1.0
        probabilities = self.model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        if confidence < MIN_CONFIDENCE:
            raise PredictionError("low_confidence", "The image could not be identified confidently. Please upload a clearer crop image.")
        label = self.labels[index] if index < len(self.labels) else "Unknown"
        mapped = CLASS_MAPPING.get(label)
        if not mapped:
            raise PredictionError("unsupported_prediction", "This image is not a supported crop or disease class.")
        return {"label": label, "crop": mapped[0], "disease": mapped[1], "confidence": confidence}


crop_model = CropDiseaseModel()
