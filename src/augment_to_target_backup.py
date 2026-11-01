
from pathlib import Path
import json
import random
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_METADATA = (
    PROJECT_DIR / "outputs" / "preprocessed" / "metadata.csv"
)
AUGMENTED_METADATA = (
    PROJECT_DIR / "outputs" / "augmented" / "augmentation_metadata.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "augmented" / "sequences"
SUMMARY_PATH = (
    PROJECT_DIR / "outputs" / "augmented" / "target_augmentation_summary.json"
)

TARGET_PER_CLASS = 350
CHECKPOINT_EVERY = 100
SEED = 123

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def resolve_original_path(value):
    """Resolve a source sequence path from the original metadata."""
    path = Path(str(value))

    if not path.is_absolute():
        path = PROJECT_DIR / "outputs" / "preprocessed" / path

    return path.resolve()


def atomic_save_csv(dataframe, destination):
    """Write metadata to a temporary file before replacing the CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    dataframe.to_csv(temporary, index=False)
    temporary.replace(destination)


def augment_sequence(sequence):
    """Apply mild transformations consistently across all frames."""
    height, width = sequence.shape[1:3]

    angle = random.uniform(-5, 5)
    tx = random.uniform(-0.025, 0.025) * width
    ty = random.uniform(-0.025, 0.025) * height

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    brightness = random.uniform(-8, 8)
    contrast = random.uniform(0.95, 1.05)

    result = []

    for frame in sequence:
        transformed = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        transformed = (
            transformed.astype(np.float32) * contrast + brightness
        )
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        result.append(transformed)

    return np.stack(result).astype(np.uint8)


def validate_sequence(path):
    """Check that a saved sequence is readable and correctly formatted."""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)

        return (
            array.shape == (12, 224, 224, 3)
            and array.dtype == np.uint8
        )

    except (OSError, ValueError, EOFError):
        return False


def save_checkpoint(records, metadata_df):
    """Persist newly generated metadata without duplicating paths."""
    if not records:
        return metadata_df

    new_df = pd.DataFrame(records)

    combined = pd.concat(
        [metadata_df, new_df],
        ignore_index=True,
    )

    combined["augmented_sequence"] = combined[
        "augmented_sequence"
    ].map(lambda p: str(Path(str(p)).resolve()))

    if combined["augmented_sequence"].duplicated().any():
        raise ValueError(
            "Duplicate augmented paths detected; metadata not saved."
        )

    atomic_save_csv(combined, AUGMENTED_METADATA)
    records.clear()

    return combined


# ============================================================
# MAIN
# ============================================================
def main():
    if not ORIGINAL_METADATA.exists():
        raise FileNotFoundError(ORIGINAL_METADATA)

    if not AUGMENTED_METADATA.exists():
        raise FileNotFoundError(AUGMENTED_METADATA)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(ORIGINAL_METADATA)
    metadata_df = pd.read_csv(AUGMENTED_METADATA)

    required_original = {
        "output_path", "label", "split", "status"
    }
    required_augmented = {
        "source_sequence",
        "augmented_sequence",
        "label",
        "split",
        "augmentation_index",
    }

    if not required_original.issubset(original_df.columns):
        raise ValueError(
            "Original metadata is missing columns: "
            f"{required_original - set(original_df.columns)}"
        )

    if not required_augmented.issubset(metadata_df.columns):
        raise ValueError(
            "Augmentation metadata is missing columns: "
            f"{required_augmented - set(metadata_df.columns)}"
        )

    # --------------------------------------------------------
    # 1. Select original training sequences only
    # --------------------------------------------------------
    train_df = original_df[
        (original_df["split"] == "train")
        & (original_df["status"] == "success")
    ].copy()

    labels = sorted(train_df["label"].unique())

    if len(labels) != 50:
        raise ValueError(
            f"Expected 50 classes, found {len(labels)}."
        )

    if metadata_df.empty:
        raise ValueError(
            "Augmentation metadata is empty. Stop and inspect the dataset."
        )

    if not (metadata_df["split"] == "train").all():
        raise ValueError(
            "Non-training records found in augmentation metadata."
        )

    if metadata_df["augmented_sequence"].duplicated().any():
        raise ValueError(
            "Duplicate paths already exist in augmentation metadata."
        )

    # --------------------------------------------------------
    # 2. Validate original sequences and build source mapping
    # --------------------------------------------------------
    source_by_stem = {}

    for _, row in train_df.iterrows():
        source = resolve_original_path(row["output_path"])

        if not source.exists():
            raise FileNotFoundError(
                f"Missing original sequence: {source}"
            )

        if not validate_sequence(source):
            raise ValueError(
                f"Invalid original sequence: {source}"
            )

        if source.stem in source_by_stem:
            raise ValueError(
                f"Duplicate source filename stem: {source.stem}"
            )

        source_by_stem[source.stem] = {
            "path": source,
            "label": row["label"],
        }

    original_counts = train_df.groupby("label").size().to_dict()

    # --------------------------------------------------------
    # 3. Validate existing augmented files
    # --------------------------------------------------------
    for value in metadata_df["augmented_sequence"]:
        path = Path(str(value))

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata points to a missing file: {path}"
            )

        if not validate_sequence(path):
            raise ValueError(
                f"Invalid existing augmented sequence: {path}"
            )

    metadata_df["augmented_sequence"] = metadata_df[
        "augmented_sequence"
    ].map(lambda p: str(Path(str(p)).resolve()))

    # Ensure each recorded augmentation refers to the correct source.
    for _, row in metadata_df.iterrows():
        source = Path(str(row["source_sequence"])).resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"Missing augmentation source: {source}"
            )

        if source.stem not in source_by_stem:
            raise ValueError(
                f"Unknown training source in augmentation metadata: {source}"
            )

        source_info = source_by_stem[source.stem]

        if source_info["path"] != source:
            raise ValueError(
                f"Source path mismatch: {source}"
            )

        if source_info["label"] != row["label"]:
            raise ValueError(
                f"Label mismatch for augmentation: {row['augmented_sequence']}"
            )

    # --------------------------------------------------------
    # 4. Recover valid files not recorded in metadata
    # --------------------------------------------------------
    known_paths = {
        str(Path(str(value)).resolve()).casefold()
        for value in metadata_df["augmented_sequence"]
    }

    recovered = []

    for file_path in OUTPUT_DIR.glob("*.npy"):
        resolved = str(file_path.resolve())

        if resolved.casefold() in known_paths:
            continue

        match = re.fullmatch(r"(.+)_aug(\d+)", file_path.stem)

        if not match:
            print(f"Skipping unrecognized file: {file_path.name}")
            continue

        source_stem, index_text = match.groups()

        if source_stem not in source_by_stem:
            print(f"Skipping unknown source: {file_path.name}")
            continue

        # Validate before recovering the record.
        if not validate_sequence(file_path):
            print(f"Skipping invalid recovered file: {file_path.name}")
            continue

        source_info = source_by_stem[source_stem]
        recovered.append({
            "source_sequence": str(source_info["path"]),
            "augmented_sequence": resolved,
            "label": source_info["label"],
            "split": "train",
            "augmentation_index": int(index_text),
        })

    if recovered:
        print(f"Recovering {len(recovered)} valid unrecorded files.")

        recovered_df = pd.DataFrame(recovered)
        metadata_df = pd.concat(
            [metadata_df, recovered_df],
            ignore_index=True,
        )

        metadata_df = metadata_df.drop_duplicates(
            subset=["augmented_sequence"],
            keep="last",
        )

        atomic_save_csv(metadata_df, AUGMENTED_METADATA)

    # --------------------------------------------------------
    # 5. Build used-index sets for unique filenames
    # --------------------------------------------------------
    used_indices = {}

    for _, row in metadata_df.iterrows():
        source = str(Path(str(row["source_sequence"])).resolve())
        used_indices.setdefault(source, set()).add(
            int(row["augmentation_index"])
        )

    original_counts = train_df.groupby("label").size().to_dict()

    records_since_checkpoint = []
    total_generated_this_run = 0

    # --------------------------------------------------------
    # 6. Generate only the missing training samples
    # --------------------------------------------------------
    for label in labels:
        originals = train_df[train_df["label"] == label]

        current_augmented = metadata_df[
            (metadata_df["label"] == label)
            & (metadata_df["split"] == "train")
        ]

        original_count = int(original_counts[label])
        augmented_count = len(current_augmented)
        current_total = original_count + augmented_count
        needed = max(0, TARGET_PER_CLASS - current_total)

        generated_for_class = 0
        source_rows = originals.to_dict("records")

        print(
            f"\n{label}: current={current_total}, "
            f"additional_needed={needed}"
        )

        while generated_for_class < needed:
            random.shuffle(source_rows)

            for row in source_rows:
                if generated_for_class >= needed:
                    break

                source_path = resolve_original_path(row["output_path"])
                source_key = str(source_path)

                sequence = np.load(
                    source_path,
                    allow_pickle=False,
                )

                if (
                    sequence.shape != (12, 224, 224, 3)
                    or sequence.dtype != np.uint8
                ):
                    raise ValueError(
                        f"Unexpected source data: {source_path}"
                    )

                indices = used_indices.setdefault(source_key, set())
                next_index = max(indices, default=0) + 1

                output_path = OUTPUT_DIR / (
                    f"{source_path.stem}_aug{next_index}.npy"
                )

                # Never overwrite an existing output.
                if output_path.exists():
                    raise FileExistsError(
                        f"Output already exists: {output_path}. "
                        "Stop and inspect metadata before continuing."
                    )

                augmented = augment_sequence(sequence)

                # Save to a temporary file, then atomically rename.
                temporary_path = output_path.with_name(
                    output_path.stem + ".tmp.npy"
                )

                np.save(temporary_path, augmented)
                temporary_path.replace(output_path)

                if not validate_sequence(output_path):
                    raise ValueError(
                        f"Saved sequence failed validation: {output_path}"
                    )

                indices.add(next_index)

                records_since_checkpoint.append({
                    "source_sequence": source_key,
                    "augmented_sequence": str(output_path.resolve()),
                    "label": label,
                    "split": "train",
                    "augmentation_index": next_index,
                })

                generated_for_class += 1
                total_generated_this_run += 1

                if len(records_since_checkpoint) >= CHECKPOINT_EVERY:
                    metadata_df = save_checkpoint(
                        records_since_checkpoint,
                        metadata_df,
                    )

                    print(
                        f"Checkpoint saved; generated this run: "
                        f"{total_generated_this_run}"
                    )

        # Checkpoint at every class boundary.
        metadata_df = save_checkpoint(
            records_since_checkpoint,
            metadata_df,
        )

        print(
            f"{label}: {current_total} -> "
            f"{current_total + generated_for_class}"
        )

        # Save current progress summary.
        current_counts = metadata_df.groupby("label").size().to_dict()

        progress_counts = {
            class_name: int(original_counts.get(class_name, 0))
            + int(current_counts.get(class_name, 0))
            for class_name in labels
        }

        SUMMARY_PATH.write_text(
            json.dumps({
                "target_per_class": TARGET_PER_CLASS,
                "classes": len(labels),
                "generated_this_run": total_generated_this_run,
                "current_training_total": sum(progress_counts.values()),
                "per_class_counts": progress_counts,
                "validation_augmented": False,
                "test_augmented": False,
                "seed": SEED,
            }, indent=2),
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # 7. Final verification
    # --------------------------------------------------------
    metadata_df = save_checkpoint(
        records_since_checkpoint,
        metadata_df,
    )

    augmented_counts = metadata_df.groupby("label").size().to_dict()

    final_counts = {
        label: int(original_counts.get(label, 0))
        + int(augmented_counts.get(label, 0))
        for label in labels
    }

    print("\n========== FINAL VERIFICATION ==========")

    for label, count in final_counts.items():
        print(f"{label}: {count}")

    total = sum(final_counts.values())

    print(f"\nTotal training samples: {total}")
    print(f"Additional generated this run: {total_generated_this_run}")

    success = all(
        count == TARGET_PER_CLASS
        for count in final_counts.values()
    )

    if success:
        print("SUCCESS: Every class has exactly 350 training samples.")
    else:
        print(
            "WARNING: Some classes have not reached the target. "
            "Inspect the summary and rerun if appropriate."
        )

    SUMMARY_PATH.write_text(
        json.dumps({
            "target_per_class": TARGET_PER_CLASS,
            "classes": len(labels),
            "additional_generated_this_run": total_generated_this_run,
            "final_total_training_samples": total,
            "per_class_final_counts": final_counts,
            "success": success,
            "validation_augmented": False,
            "test_augmented": False,
            "seed": SEED,
        }, indent=2),
        encoding="utf-8",
    )

    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
