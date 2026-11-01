
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_DIR / "outputs" / "vgg19_features"
MODEL_DIR = PROJECT_DIR / "models" / "transformer"

MODEL_PATH = MODEL_DIR / "best_transformer.keras"
METADATA_PATH = FEATURES_DIR / "metadata.csv"
LABELS_PATH = MODEL_DIR / "label_classes.json"

NUM_FRAMES = 12
FEATURE_DIM = 512
EXPECTED_TEST_SAMPLES = 200


# ============================================================
# LOAD TEST FEATURES
# ============================================================

def load_test_data(metadata):
    rows = metadata[
        metadata["split"].astype(str).str.strip().str.lower()
        == "test"
    ].copy()

    if len(rows) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, "
            f"found {len(rows)}"
        )

    features = []
    labels = []

    for _, row in rows.iterrows():
        path = FEATURES_DIR / str(row["feature_path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        x = np.load(path, allow_pickle=False)

        if x.shape != (NUM_FRAMES, FEATURE_DIM):
            raise ValueError(
                f"Invalid feature shape {x.shape}: {path}"
            )

        if x.dtype != np.float32:
            raise ValueError(
                f"Invalid dtype {x.dtype}: {path}"
            )

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite features: {path}")

        features.append(x)
        labels.append(str(row["label"]).strip())

    return np.stack(features), np.asarray(labels), rows


# ============================================================
# EVALUATION
# ============================================================

def main():
    for path in (MODEL_PATH, METADATA_PATH, LABELS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(METADATA_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        saved_classes = json.load(file)

    # Use the exact class order saved during training.
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(saved_classes)

    X_test, y_test_text, test_rows = load_test_data(metadata)

    unseen = sorted(set(y_test_text) - set(saved_classes))
    if unseen:
        raise ValueError(
            f"Test labels missing from training class mapping: {unseen}"
        )

    y_test = label_encoder.transform(y_test_text)

    positions_test = np.tile(
        np.arange(NUM_FRAMES, dtype=np.int32),
        (len(X_test), 1),
    )

    print("=" * 60)
    print("TRANSFORMER FINAL TEST EVALUATION")
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("Model:", MODEL_PATH)
    print("Test features:", X_test.shape)
    print("Test samples:", len(X_test))
    print("Classes:", len(saved_classes))

    print("\nLoading best checkpoint...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_loss, test_accuracy = model.evaluate(
        [X_test, positions_test],
        y_test,
        batch_size=16,
        verbose=1,
    )

    probabilities = model.predict(
        [X_test, positions_test],
        batch_size=16,
        verbose=1,
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    report_text = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(saved_classes)),
        target_names=saved_classes,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(saved_classes)),
        target_names=saved_classes,
        output_dict=True,
        zero_division=0,
    )

    print("\nTest loss:", round(float(test_loss), 4))
    print("Test accuracy:", f"{accuracy:.4%}")
    print("\nClassification report:\n")
    print(report_text)

    # Save per-video predictions for error analysis.
    prediction_rows = test_rows.copy()
    prediction_rows["true_label"] = y_test_text
    prediction_rows["predicted_label"] = (
        label_encoder.inverse_transform(predictions)
    )
    prediction_rows["correct"] = (
        prediction_rows["true_label"]
        == prediction_rows["predicted_label"]
    )
    prediction_rows["confidence"] = np.max(
        probabilities, axis=1
    )

    prediction_path = MODEL_DIR / "test_predictions.csv"
    prediction_rows.to_csv(prediction_path, index=False)

    # Save metrics.
    metrics = {
        "model": str(MODEL_PATH),
        "test_samples": len(X_test),
        "num_classes": len(saved_classes),
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "classification_report": report_dict,
    }

    metrics_path = MODEL_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Confusion matrix.
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(saved_classes)),
    )

    fig, ax = plt.subplots(figsize=(20, 18))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=saved_classes,
    )
    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("VGG19 + Transformer — Test Confusion Matrix")
    fig.tight_layout()

    cm_path = MODEL_DIR / "test_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Show the most frequently confused class pairs.
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)

    pair_indices = np.dstack(
        np.unravel_index(
            np.argsort(off_diagonal.ravel())[::-1],
            off_diagonal.shape,
        )
    )[0]

    print("\nMost confused class pairs:")
    shown = 0
    for true_idx, pred_idx in pair_indices:
        count = int(off_diagonal[true_idx, pred_idx])
        if count == 0:
            break

        print(
            f"Actual {saved_classes[true_idx]} -> "
            f"Predicted {saved_classes[pred_idx]}: {count}"
        )
        shown += 1
        if shown == 10:
            break

    print("\nSaved results:")
    print("Metrics:", metrics_path)
    print("Predictions:", prediction_path)
    print("Confusion matrix:", cm_path)
    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
