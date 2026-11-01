
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_DIR / "outputs" / "preprocessed"
FEATURES_DIR = PROJECT_DIR / "outputs" / "vgg19_features"

EXPECTED_FRAME_SHAPE = (12, 224, 224, 3)
FEATURE_DIM = 512
BATCH_SIZE = 8


def extract_features(model, sequence):
    """Extract one 512-dimensional feature vector per frame."""

    if sequence.shape != EXPECTED_FRAME_SHAPE:
        raise ValueError(
            f"Expected {EXPECTED_FRAME_SHAPE}, got {sequence.shape}"
        )

    # VGG19 expects float32 input processed using its own preprocessing.
    sequence = sequence.astype(np.float32)
    sequence = preprocess_input(sequence)

    features = model.predict(
        sequence,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    expected_shape = (12, FEATURE_DIM)

    if features.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected feature shape: {features.shape}"
        )

    return features.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N videos for a test run.",
    )
    args = parser.parse_args()

    metadata_path = PREPROCESSED_DIR / "metadata.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        metadata = metadata.head(args.limit).copy()

    for split in ("train", "val", "test"):
        (FEATURES_DIR / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Loading pretrained VGG19...")
    print("TensorFlow version:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

    # Global average pooling produces 512 features per frame.
    model = VGG19(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    model.trainable = False

    successful = []
    failed = []

    print("\nVideos to process:", len(metadata))

    for count, (_, row) in enumerate(
        metadata.iterrows(), start=1
    ):
        split = str(row["split"]).strip().lower()
        source_path = str(row["source_path"])
        label = str(row["label"])
        csv_row_index = int(row["csv_row_index"])

        input_path = PREPROCESSED_DIR / str(row["output_path"])
        output_path = (
            FEATURES_DIR
            / split
            / f"row_{csv_row_index:06d}.npy"
        )

        print(
            f"[{count}/{len(metadata)}] "
            f"{label} | {split}"
        )

        try:
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Missing preprocessed array: {input_path}"
                )

            sequence = np.load(
                input_path,
                allow_pickle=False,
            )

            features = extract_features(model, sequence)

            np.save(output_path, features)

            # Verify saved feature sequence.
            saved = np.load(
                output_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if saved.shape != (12, FEATURE_DIM):
                raise RuntimeError(
                    f"Wrong saved feature shape: {saved.shape}"
                )

            if saved.dtype != np.float32:
                raise RuntimeError(
                    f"Wrong saved dtype: {saved.dtype}"
                )

            successful.append({
                "csv_row_index": csv_row_index,
                "source_path": source_path,
                "label": label,
                "split": split,
                "feature_path": str(
                    output_path.relative_to(FEATURES_DIR)
                ),
                "num_frames": 12,
                "feature_dim": FEATURE_DIM,
                "dtype": "float32",
                "status": "success",
            })

            print(
                f"SUCCESS | shape={saved.shape} "
                f"| dtype={saved.dtype}"
            )

        except Exception as error:
            failed.append({
                "csv_row_index": csv_row_index,
                "source_path": source_path,
                "label": label,
                "split": split,
                "error": str(error),
            })
            print("FAILED:", error)

    # Write logs for this run.
    pd.DataFrame(successful).to_csv(
        FEATURES_DIR / "metadata.csv",
        index=False,
    )

    pd.DataFrame(
        failed,
        columns=[
            "csv_row_index",
            "source_path",
            "label",
            "split",
            "error",
        ],
    ).to_csv(
        FEATURES_DIR / "failed_features.csv",
        index=False,
    )

    summary = {
        "input_metadata": str(metadata_path),
        "feature_directory": str(FEATURES_DIR),
        "method": "VGG19 ImageNet pretrained, frozen",
        "pooling": "global_average",
        "feature_dim": FEATURE_DIM,
        "frames_per_video": 12,
        "input_videos_this_run": len(metadata),
        "successful": len(successful),
        "failed": len(failed),
        "expected_feature_shape": [12, FEATURE_DIM],
        "dtype": "float32",
        "run_limit": args.limit,
    }

    with open(
        FEATURES_DIR / "feature_extraction_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION FINISHED")
    print("Input videos:", len(metadata))
    print("Successful:", len(successful))
    print("Failed:", len(failed))
    print("Output:", FEATURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
