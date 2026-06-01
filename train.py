"""Train a transfer-learning model on the food-mini dataset by default."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import (
    CLASS_LABELS_PATH,
    DATASET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    RANDOM_SEED,
    ensure_project_directories,
    get_class_names,
    save_class_labels,
    save_training_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a food image classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to dataset class folders.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze upper base-model layers after warmup.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Extra epochs for fine-tuning.")
    parser.add_argument("--fine-tune-at", type=int, default=100, help="Layer index from which to unfreeze base model.")
    return parser.parse_args()


def configure_gpu() -> None:
    """Enable memory growth when a compatible GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if gpus:
        print(f"GPU detected: {len(gpus)} device(s)")
    else:
        print("No GPU detected. Training will run on CPU.")


def create_data_generators(dataset_dir: Path, batch_size: int, validation_split: float):
    """Create train and validation generators with augmentation."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=validation_split,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.12,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )

    validation_generator = validation_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )

    return train_generator, validation_generator


def build_model(num_classes: int, learning_rate: float) -> tf.keras.Model:
    """Build MobileNetV2 transfer-learning model."""
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="food_classifier_mobilenetv2")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for stable and efficient training."""
    ensure_project_directories()
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def fine_tune_model(
    model: tf.keras.Model,
    train_generator,
    validation_generator,
    fine_tune_at: int,
    fine_tune_epochs: int,
) -> tf.keras.callbacks.History:
    """Optionally fine-tune upper MobileNetV2 layers."""
    base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=fine_tune_epochs,
        callbacks=create_callbacks(),
    )


def main() -> None:
    args = parse_args()
    ensure_project_directories()
    configure_gpu()

    class_names = get_class_names(args.dataset_dir)
    print(f"Found {len(class_names)} classes.")

    train_generator, validation_generator = create_data_generators(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    index_to_class = {
        index: class_name for class_name, index in train_generator.class_indices.items()
    }
    ordered_class_names = [index_to_class[index] for index in range(len(index_to_class))]
    save_class_labels(ordered_class_names, CLASS_LABELS_PATH)
    print(f"Saved class labels to {CLASS_LABELS_PATH}")

    model = build_model(num_classes=len(ordered_class_names), learning_rate=args.learning_rate)
    model.summary()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        callbacks=create_callbacks(),
    )

    if args.fine_tune:
        print("Starting fine-tuning...")
        history = fine_tune_model(
            model=model,
            train_generator=train_generator,
            validation_generator=validation_generator,
            fine_tune_at=args.fine_tune_at,
            fine_tune_epochs=args.fine_tune_epochs,
        )

    model.save(MODEL_PATH)
    save_training_plots(history)
    print(f"Training complete. Model saved to {MODEL_PATH}")
    print("Training graphs saved to outputs/accuracy.png and outputs/loss.png")


if __name__ == "__main__":
    main()
