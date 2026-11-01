
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model, layers, callbacks


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_DIR / "outputs" / "vgg19_features"
MODEL_DIR = PROJECT_DIR / "models" / "transformer"

SEED = 42
NUM_FRAMES = 12
FEATURE_DIM = 512
D_MODEL = 128
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.2
BATCH_SIZE = 16
EPOCHS = 50

EXPECTED_SPLITS = {
    "train": 700,
    "val": 150,
    "test": 200,
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD AND VALIDATE FEATURES
# ============================================================

def load_split(metadata, split):
    rows = metadata[metadata["split"] == split].copy()

    features = []
    labels = []

    for _, row in rows.iterrows():
        feature_path = FEATURES_DIR / str(row["feature_path"])

        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)

        x = np.load(feature_path, allow_pickle=False)

        if x.shape != (NUM_FRAMES, FEATURE_DIM):
            raise ValueError(
                f"Invalid shape {x.shape}: {feature_path}"
            )

        if x.dtype != np.float32:
            raise ValueError(
                f"Invalid dtype {x.dtype}: {feature_path}"
            )

        if not np.isfinite(x).all():
            raise ValueError(
                f"Non-finite values in {feature_path}"
            )

        features.append(x)
        labels.append(str(row["label"]).strip())

    if not features:
        raise ValueError(f"No feature sequences found for {split}")

    return np.stack(features), np.asarray(labels)


# ============================================================
# 3. TRANSFORMER ENCODER BLOCK
# ============================================================

def transformer_encoder(x, block_number):
    # Multi-head self-attention across the 12 frames.
    attention = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=D_MODEL // NUM_HEADS,
        dropout=DROPOUT,
        name=f"attention_{block_number}",
    )(x, x)

    attention = layers.Dropout(DROPOUT)(attention)
    x = layers.LayerNormalization(epsilon=1e-6)(
        x + attention
    )

    # Feed-forward network applied to each frame position.
    ff = layers.Dense(
        FF_DIM,
        activation="relu",
        name=f"ff_expand_{block_number}",
    )(x)

    ff = layers.Dropout(DROPOUT)(ff)
    ff = layers.Dense(
        D_MODEL,
        name=f"ff_project_{block_number}",
    )(ff)

    x = layers.LayerNormalization(epsilon=1e-6)(
        x + ff
    )

    return x


# ============================================================
# 4. BUILD MODEL
# ============================================================

def build_transformer(num_classes):
    feature_input = layers.Input(
        shape=(NUM_FRAMES, FEATURE_DIM),
        name="frame_features",
    )

    # Explicit positional embeddings let attention distinguish
    # early, middle, and late frames.
    positions_input = layers.Input(
        shape=(NUM_FRAMES,),
        dtype="int32",
        name="frame_positions",
    )

    x = layers.Dense(
        D_MODEL,
        name="feature_projection",
    )(feature_input)

    position_embeddings = layers.Embedding(
        input_dim=NUM_FRAMES,
        output_dim=D_MODEL,
        name="position_embedding",
    )(positions_input)

    x = layers.Add(name="add_positions")(
        [x, position_embeddings]
    )

    x = layers.Dropout(DROPOUT)(x)

    # Two Transformer encoder blocks.
    x = transformer_encoder(x, block_number=1)
    x = transformer_encoder(x, block_number=2)

    # Aggregate information across the temporal sequence.
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)

    output = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = Model(
        inputs=[feature_input, positions_input],
        outputs=output,
        name="VGG19_Transformer_SLR",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = FEATURES_DIR / "metadata.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "csv_row_index",
        "label",
        "split",
        "feature_path",
        "status",
    }

    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    metadata["split"] = (
        metadata["split"].astype(str).str.strip().str.lower()
    )
    metadata["label"] = metadata["label"].astype(str).str.strip()

    if metadata["csv_row_index"].duplicated().any():
        raise ValueError("Duplicate source row IDs in feature metadata.")

    if len(metadata) != 1050:
        raise ValueError(
            f"Expected 1050 metadata rows, found {len(metadata)}"
        )

    if not (metadata["status"] == "success").all():
        raise ValueError(
            "Feature metadata contains unsuccessful records."
        )

    if metadata["feature_path"].duplicated().any():
        raise ValueError("Duplicate feature paths in metadata.")

    for split, expected_count in EXPECTED_SPLITS.items():
        actual_count = int((metadata["split"] == split).sum())
        if actual_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count}, "
                f"found {actual_count}"
            )

    classes = sorted(metadata["label"].unique().tolist())

    if len(classes) != 50:
        raise ValueError(
            f"Expected 50 classes, found {len(classes)}"
        )

    # Fit the label mapping using the training labels only.
    X_train, y_train_text = load_split(metadata, "train")
    X_val, y_val_text = load_split(metadata, "val")
    X_test, y_test_text = load_split(metadata, "test")

    encoder = LabelEncoder()
    encoder.fit(y_train_text)

    known_classes = set(encoder.classes_)

    for split_name, labels in [
        ("validation", y_val_text),
        ("test", y_test_text),
    ]:
        unseen = sorted(set(labels) - known_classes)
        if unseen:
            raise ValueError(
                f"Classes absent from training but present in "
                f"{split_name}: {unseen}"
            )

    y_train = encoder.transform(y_train_text)
    y_val = encoder.transform(y_val_text)
    y_test = encoder.transform(y_test_text)

    # Position indices: 0 through 11 for each sequence.
    positions = np.tile(
        np.arange(NUM_FRAMES, dtype=np.int32),
        (1, 1),
    )

    positions_train = np.repeat(positions, len(X_train), axis=0)
    positions_val = np.repeat(positions, len(X_val), axis=0)
    positions_test = np.repeat(positions, len(X_test), axis=0)

    print("=" * 60)
    print("VGG19 + TRANSFORMER SIGN LANGUAGE CLASSIFIER")
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("Classes:", len(encoder.classes_))
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    model = build_transformer(
        num_classes=len(encoder.classes_)
    )

    model.summary()

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_transformer.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(
            str(MODEL_DIR / "training_history.csv")
        ),
    ]

    history = model.fit(
        [X_train, positions_train],
        y_train,
        validation_data=(
            [X_val, positions_val],
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        shuffle=True,
        verbose=1,
    )

    # Save the final model and class mapping.
    model.save(MODEL_DIR / "final_transformer.keras")

    with open(
        MODEL_DIR / "label_classes.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            encoder.classes_.tolist(),
            file,
            indent=2,
        )

    with open(
        MODEL_DIR / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "architecture": "VGG19 frozen features + Transformer",
                "num_classes": len(encoder.classes_),
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "epochs_completed": len(history.history["loss"]),
                "best_validation_accuracy": float(
                    max(history.history["val_accuracy"])
                ),
                "best_validation_loss": float(
                    min(history.history["val_loss"])
                ),
                "test_evaluation": "Not yet performed",
            },
            file,
            indent=2,
        )

    print("\nTraining finished.")
    print("Best checkpoint:", MODEL_DIR / "best_transformer.keras")
    print("Final model:", MODEL_DIR / "final_transformer.keras")
    print("Test set remains reserved for final evaluation.")


if __name__ == "__main__":
    main()
