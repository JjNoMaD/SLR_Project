
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_DIR / "outputs" / "preprocessed" / "metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "augmented" / "sequences"
)

NUM_AUGMENTATIONS = 2
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


def augment_sequence(sequence):
    """Apply consistent transformations across all frames."""

    num_frames, height, width, channels = sequence.shape

    angle = float(rng.uniform(-5, 5))
    tx = float(rng.uniform(-0.03, 0.03) * width)
    ty = float(rng.uniform(-0.03, 0.03) * height)

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    alpha = float(rng.uniform(0.90, 1.10))
    beta = float(rng.uniform(-10, 10))

    result = np.empty_like(sequence)

    for i in range(num_frames):
        frame = cv2.warpAffine(
            sequence[i],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        frame = np.clip(
            frame.astype(np.float32) * alpha + beta,
            0,
            255,
        ).astype(np.uint8)

        result[i] = frame

    return result


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    required = {
        "label", "split", "output_path", "status"
    }
    missing_columns = required - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    train = metadata[
        (metadata["split"] == "train")
        & (metadata["status"] == "success")
    ].copy()

    if len(train) != 700:
        raise ValueError(
            f"Expected 700 training sequences; found {len(train)}"
        )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_DIR}\n"
            "Inspect it before rerunning; no files were overwritten."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []

    for count, row in enumerate(train.itertuples(index=False), 1):
        source = Path(row.output_path)

        if not source.is_absolute():
            source = PROJECT_DIR / "outputs" / "preprocessed" / source

        try:
            if not source.is_file():
                raise FileNotFoundError(source)

            sequence = np.load(source, allow_pickle=False)

            if sequence.shape != (12, 224, 224, 3):
                raise ValueError(
                    f"Unexpected shape: {sequence.shape}"
                )

            if sequence.dtype != np.uint8:
                raise ValueError(
                    f"Unexpected dtype: {sequence.dtype}"
                )

            for aug_index in range(1, NUM_AUGMENTATIONS + 1):
                augmented = augment_sequence(sequence)

                output_name = (
                    f"{source.stem}_aug{aug_index}.npy"
                )
                output_path = OUTPUT_DIR / output_name

                np.save(output_path, augmented)

                records.append({
                    "source_sequence": str(source),
                    "augmented_sequence": str(output_path),
                    "label": row.label,
                    "split": "train",
                    "augmentation_index": aug_index,
                })

        except Exception as exc:
            failures.append({
                "source_sequence": str(source),
                "label": row.label,
                "error": str(exc),
            })

        if count % 50 == 0:
            print(f"Processed {count}/{len(train)} originals")

    metadata_out = OUTPUT_DIR.parent / "augmentation_metadata.csv"
    failures_out = OUTPUT_DIR.parent / "augmentation_failures.csv"
    summary_out = OUTPUT_DIR.parent / "augmentation_summary.json"

    pd.DataFrame(records).to_csv(metadata_out, index=False)
    pd.DataFrame(failures).to_csv(failures_out, index=False)

    summary = {
        "original_training_sequences": len(train),
        "copies_per_sequence": NUM_AUGMENTATIONS,
        "augmented_sequences_saved": len(records),
        "failed_source_sequences": len(failures),
        "validation_augmented": False,
        "test_augmented": False,
        "horizontal_flip_used": False,
        "seed": RANDOM_SEED,
    }

    summary_out.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== AUGMENTATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if failures:
        raise RuntimeError(
            f"{len(failures)} source sequences failed. "
            f"Inspect {failures_out}"
        )


if __name__ == "__main__":
    main()
