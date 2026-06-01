"""Shared utilities for the food image classification project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "food-mini"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "food_model.h5"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.json"

IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
RANDOM_SEED = 42

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_project_directories() -> None:
    """Create generated-output directories if they do not exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_class_names(dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Read class names from dataset folder names."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Place mini dataset class folders inside dataset/food-mini/."
        )

    class_names = sorted(
        item.name for item in dataset_dir.iterdir() if item.is_dir() and not item.name.startswith(".")
    )

    if not class_names:
        raise ValueError(
            f"No class folders found in {dataset_dir}. "
            "Expected folders like pizza, sushi, apple_pie, etc."
        )

    return class_names


def save_class_labels(class_names: Iterable[str], output_path: Path = CLASS_LABELS_PATH) -> None:
    """Save class labels in prediction-index order."""
    ensure_project_directories()
    output_path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")


def load_class_labels(labels_path: Path = CLASS_LABELS_PATH) -> list[str]:
    """Load class labels saved during training."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Class labels file not found: {labels_path}. "
            "Train the model first so class_labels.json is created."
        )
    return json.loads(labels_path.read_text(encoding="utf-8"))


def format_label(label: str) -> str:
    """Convert folder-style class names into readable names."""
    return label.replace("_", " ").title()


def validate_image_file(file_obj) -> Image.Image:
    """Open and verify an uploaded image-like object."""
    try:
        image = Image.open(file_obj)
        image.verify()
        file_obj.seek(0)
        image = Image.open(file_obj).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image


def preprocess_pil_image(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize and normalize a PIL image for MobileNetV2 inference."""
    image = image.convert("RGB").resize(image_size)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1.0
    return np.expand_dims(image_array, axis=0)


def save_training_plots(history, output_dir: Path = OUTPUTS_DIR) -> None:
    """Save training accuracy and loss charts as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dict = history.history

    plot_metric(
        history_dict,
        train_metric="accuracy",
        validation_metric="val_accuracy",
        title="Model Accuracy",
        ylabel="Accuracy",
        output_path=output_dir / "accuracy.png",
    )
    plot_metric(
        history_dict,
        train_metric="loss",
        validation_metric="val_loss",
        title="Model Loss",
        ylabel="Loss",
        output_path=output_dir / "loss.png",
    )


def plot_metric(
    history_dict: dict,
    train_metric: str,
    validation_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a single train/validation metric chart."""
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict.get(train_metric, []), label=f"Training {ylabel}")
    plt.plot(history_dict.get(validation_metric, []), label=f"Validation {ylabel}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def prediction_result_to_json(result: dict) -> str:
    """Serialize prediction output for downloads or logs."""
    return json.dumps(result, indent=2)
