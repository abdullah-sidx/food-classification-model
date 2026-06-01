"""Reusable prediction helpers for Food-mini image classification."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import (
    CLASS_LABELS_PATH,
    MODEL_PATH,
    format_label,
    load_class_labels,
    preprocess_pil_image,
    validate_image_file,
)


def load_trained_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load the saved Keras model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with: python train.py"
        )
    return tf.keras.models.load_model(model_path)


def predict_image(
    image: Image.Image,
    model: tf.keras.Model | None = None,
    class_names: list[str] | None = None,
    top_k: int = 3,
) -> dict:
    """Predict the food class for a PIL image."""
    model = model or load_trained_model()
    class_names = class_names or load_class_labels(CLASS_LABELS_PATH)

    image_batch = preprocess_pil_image(image)
    probabilities = model.predict(image_batch, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "class_name": class_names[index],
            "display_name": format_label(class_names[index]),
            "confidence": float(probabilities[index]),
            "confidence_percent": round(float(probabilities[index]) * 100, 2),
        }
        for index in top_indices
    ]

    best_prediction = top_predictions[0]
    return {
        "predicted_class": best_prediction["class_name"],
        "display_name": best_prediction["display_name"],
        "confidence": best_prediction["confidence"],
        "confidence_percent": best_prediction["confidence_percent"],
        "top_predictions": top_predictions,
    }


def predict_uploaded_file(
    file_obj: BinaryIO,
    model: tf.keras.Model | None = None,
    class_names: list[str] | None = None,
    top_k: int = 3,
) -> tuple[Image.Image, dict]:
    """Validate an uploaded file and return its image plus prediction."""
    image = validate_image_file(file_obj)
    result = predict_image(image=image, model=model, class_names=class_names, top_k=top_k)
    return image, result


def predict_from_path(image_path: Path, top_k: int = 3) -> dict:
    """Predict from a local image path, useful for scripts or tests."""
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError("Unsupported image extension.")

    with Image.open(image_path) as image:
        return predict_image(image.convert("RGB"), top_k=top_k)

import sys
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        exit()

    image_path = Path(sys.argv[1])

    result = predict_from_path(image_path)

    print("\n--- PREDICTION RESULT ---")
    print("Class:", result["predicted_class"])
    print("Confidence:", f"{result['confidence_percent']}%")

    print("\nTop predictions:")
    for p in result["top_predictions"]:
        print(f"- {p['display_name']}: {p['confidence_percent']}%")