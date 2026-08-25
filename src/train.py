"""
train.py
--------
M1 Task 2/3: baseline CNN + experiment tracking.

Loads the already-preprocessed dataset (data/processed/{train,val,test}/{cat,dog}/,
see preprocess.py), trains a small CNN with on-the-fly augmentation, logs the run
to MLflow (params, metrics, loss curves, confusion matrix), and saves the fitted
model to models/cats_dogs_cnn.keras + a metadata sidecar for the API.

Two ways to run this:
  - Locally, for the real artifact used by the API:
        python src/train.py
  - In CI, as a fast smoke test that only proves the pipeline works end-to-end
    (tiny sample, one epoch), using env vars:
        SAMPLE_SIZE=10 EPOCHS=1 python src/train.py --data tests/fixtures/sample_images

Then inspect:
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # -> http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "models" / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
RANDOM_STATE = 42
CLASS_NAMES = ["cat", "dog"]  # alphabetical, matches image_dataset_from_directory


def build_datasets(data_dir: Path, batch_size: int, sample_size: int | None):
    """Load train/val/test splits from data_dir/{split}/{cat,dog}/."""
    def _load(split):
        # Always shuffle (with a fixed seed, for reproducibility): directory
        # listing groups all "cat" files before all "dog" files, so an
        # unshuffled `.take()` below would silently grab a single-class
        # prefix instead of a representative sample.
        ds = tf.keras.utils.image_dataset_from_directory(
            data_dir / split,
            labels="inferred",
            label_mode="binary",
            class_names=CLASS_NAMES,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            shuffle=True,
            seed=RANDOM_STATE,
        )
        if sample_size is not None:
            ds = ds.take(max(1, sample_size // batch_size))
        return ds

    train_ds = _load("train")
    val_ds = _load("val")
    test_ds = _load("test")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds


def build_model() -> tf.keras.Model:
    """A simple baseline CNN: 4 conv blocks + GAP + dense head. Not transfer learning."""
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 128):
        x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cats_dogs_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def log_loss_curves(history) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    path = FIGS / "loss_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def log_confusion_matrix(y_true, y_pred) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix — cats_dogs_cnn")
    fig.tight_layout()
    path = FIGS / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 8)))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sample-size", type=int,
        default=(int(os.environ["SAMPLE_SIZE"]) if "SAMPLE_SIZE" in os.environ else None),
        help="Cap images per split (fast CI smoke runs); default uses the full processed set.",
    )
    parser.add_argument(
        "--model-out", type=Path, default=None,
        help="Override models/cats_dogs_cnn.keras (e.g. a scratch path for CI smoke runs "
             "so the real, git-committed model isn't clobbered).",
    )
    parser.add_argument("--meta-out", type=Path, default=None)
    args = parser.parse_args()

    model_out = args.model_out or (MODELS / "cats_dogs_cnn.keras")
    meta_out = args.meta_out or (MODELS / "model_metadata.json")
    model_out.parent.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(RANDOM_STATE)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    os.makedirs(ROOT / "mlartifacts", exist_ok=True)
    mlflow.set_experiment("cats-dogs-classification")
    mlflow.tensorflow.autolog(log_models=False)  # we log the model ourselves below

    train_ds, val_ds, test_ds = build_datasets(args.data, args.batch_size, args.sample_size)
    model = build_model()

    with mlflow.start_run(run_name="baseline_cnn"):
        mlflow.log_params({
            "img_size": IMG_SIZE,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "sample_size": args.sample_size,
            "architecture": "conv32-64-128-128+gap+dense128",
        })

        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc, "test_auc": test_auc})

        y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int).ravel()
        y_prob = model.predict(test_ds, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        loss_path = log_loss_curves(history)
        cm_path = log_confusion_matrix(y_true, y_pred)
        mlflow.log_artifact(str(loss_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        report_path = MODELS / "classification_report.txt"
        report_path.write_text(report_txt)
        mlflow.log_artifact(str(report_path))
        print(report_txt)

        mlflow.tensorflow.log_model(model, artifact_path="model")

        model.save(model_out)

        meta = {
            "img_size": IMG_SIZE,
            "class_names": CLASS_NAMES,
            "metrics": {
                "test_loss": round(float(test_loss), 4),
                "test_accuracy": round(float(test_acc), 4),
                "test_auc": round(float(test_auc), 4),
            },
            "epochs": args.epochs,
            "sample_size": args.sample_size,
        }
        meta_out.write_text(json.dumps(meta, indent=2))

        print(f"Saved model -> {model_out}")
        print(f"Saved metadata -> {meta_out}")
        print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
