
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent.parent

AUGMENTED_METADATA = (
    PROJECT_DIR / "outputs" / "augmented"
    / "augmentation_metadata.csv"
)

FEATURES_DIR = (
    PROJECT_DIR / "outputs" / "augmented"
    / "vgg19_features"
)

FEATURE_METADATA = FEATURES_DIR / "metadata.csv"

EXPECTED_SHAPE = (12, 224, 224, 3)
FEATURE_SHAPE = (12, 512)
BATCH_SIZE = 8

COLUMNS = [
    "source_sequence",
    "augmented_sequence",
    "label",
    "split",
    "augmentation_index",
    "feature_path",
    "status",
]


def extract_features(model, sequence):
    if sequence.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected {EXPECTED_SHAPE}, got {sequence.shape}"
        )

    if sequence.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8, got {sequence.dtype}"
        )

    frames = preprocess_input(
        sequence.astype(np.float32)
    )

    features = model.predict(
        frames,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    if features.shape != FEATURE_SHAPE:
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    return features.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending sequences.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    if not AUGMENTED_METADATA.exists():
        raise FileNotFoundError(AUGMENTED_METADATA)

    metadata = pd.read_csv(AUGMENTED_METADATA)

    if len(metadata) != 16800:
        raise ValueError(
            f"Expected 16800 augmented sequences, got {len(metadata)}"
        )

    if not (metadata["split"] == "train").all():
        raise ValueError(
            "Non-training records found in augmented metadata."
        )

    if metadata["augmented_sequence"].duplicated().any():
        raise ValueError(
            "Duplicate augmented paths detected."
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURE_METADATA.exists():
        completed = pd.read_csv(FEATURE_METADATA)
        if not set(COLUMNS).issubset(completed.columns):
            raise ValueError(
                "Existing feature metadata has unexpected columns."
            )
    else:
        completed = pd.DataFrame(columns=COLUMNS)

    completed_paths = set(
        completed.loc[
            completed["status"] == "success",
            "augmented_sequence",
        ].astype(str)
    )

    pending = metadata[
        ~metadata["augmented_sequence"].astype(str).isin(
            completed_paths
        )
    ].copy()

    if args.limit is not None:
        pending = pending.head(args.limit)

    print("Total augmented sequences:", len(metadata))
    print("Already completed:", len(completed_paths))
    print("Pending this run:", len(pending))

    if pending.empty:
        print("No pending sequences.")
        return

    print("TensorFlow version:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("Loading frozen VGG19...")

    model = VGG19(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    model.trainable = False

    processed = 0
    failed = []

    for _, row in pending.iterrows():
        source_path = Path(str(row["augmented_sequence"]))

        # Use the original source's unique row ID plus
        # augmentation index to avoid filename collisions.
        source_stem = Path(str(row["source_sequence"])).stem
        augmentation_index = int(row["augmentation_index"])

        output_path = FEATURES_DIR / (
            f"{source_stem}_aug{augmentation_index}.npy"
        )

        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            if output_path.exists():
                existing = np.load(
                    output_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if (
                    existing.shape != FEATURE_SHAPE
                    or existing.dtype != np.float32
                ):
                    raise ValueError(
                        f"Invalid existing feature file: {output_path}"
                    )
            else:
                sequence = np.load(
                    source_path,
                    allow_pickle=False,
                )

                features = extract_features(model, sequence)

                temporary = output_path.with_name(
                    output_path.stem + ".tmp.npy"
                )

                np.save(temporary, features)
                temporary.replace(output_path)

            completed = pd.concat([
                completed,
                pd.DataFrame([{
                    "source_sequence": str(row["source_sequence"]),
                    "augmented_sequence": str(source_path),
                    "label": str(row["label"]),
                    "split": "train",
                    "augmentation_index": augmentation_index,
                    "feature_path": str(output_path.resolve()),
                    "status": "success",
                }]),
            ], ignore_index=True)

            processed += 1

            if processed % 100 == 0:
                temporary_csv = FEATURE_METADATA.with_suffix(
                    ".csv.tmp"
                )
                completed.to_csv(temporary_csv, index=False)
                temporary_csv.replace(FEATURE_METADATA)

                print(
                    f"Checkpoint: {processed}/{len(pending)} "
                    f"processed this run."
                )

        except Exception as error:
            failed.append({
                "augmented_sequence": str(source_path),
                "label": str(row["label"]),
                "error": str(error),
            })

            print("FAILED:", source_path.name, error)

    temporary_csv = FEATURE_METADATA.with_suffix(".csv.tmp")
    completed.to_csv(temporary_csv, index=False)
    temporary_csv.replace(FEATURE_METADATA)

    pd.DataFrame(
        failed,
        columns=["augmented_sequence", "label", "error"],
    ).to_csv(
        FEATURES_DIR / "failed_features.csv",
        index=False,
    )

    summary = {
        "total_augmented_sequences": len(metadata),
        "completed_feature_records": len(completed),
        "processed_this_run": processed,
        "failed_this_run": len(failed),
        "feature_shape": list(FEATURE_SHAPE),
        "dtype": "float32",
        "model": "Frozen VGG19 ImageNet",
        "pooling": "global_average",
        "baseline_features_modified": False,
    }

    with open(
        FEATURES_DIR / "feature_extraction_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nFeature extraction run finished.")
    print("Processed this run:", processed)
    print("Failed this run:", len(failed))
    print("Completed records:", len(completed))
    print("Output directory:", FEATURES_DIR)


if __name__ == "__main__":
    main()
