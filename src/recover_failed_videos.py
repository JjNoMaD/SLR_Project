
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# Make the src directory importable when run from the project root.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from preprocess_dataset import (
    CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    TARGET_FRAMES,
    IMG_SIZE,
    preprocess_video,
    resolve_video_path,
)

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FAILED_PATH = OUTPUT_DIR / "failed_videos.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
SUMMARY_PATH = OUTPUT_DIR / "preprocessing_summary.json"


def main():
    if not FAILED_PATH.is_file():
        raise FileNotFoundError(
            f"Failure log not found: {FAILED_PATH}"
        )

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    failures = pd.read_csv(FAILED_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if failures.empty:
        print("No failed videos remain. Nothing to recover.")
        return

    recovered_records = []
    remaining_failures = []

    print("=" * 65)
    print("RECOVERING FAILED VIDEOS")
    print("Videos to retry:", len(failures))
    print("Output directory:", OUTPUT_DIR)
    print("=" * 65)

    for count, (_, row) in enumerate(
        failures.iterrows(), start=1
    ):
        source_path = resolve_video_path(row["source_path"])
        label = str(row["label"])
        split = str(row["split"]).strip().lower()
        csv_row_index = int(row["csv_row_index"])

        output_name = f"row_{csv_row_index:06d}.npy"
        output_path = (
            OUTPUT_DIR / "sequences" / split / output_name
        )

        print(
            f"\n[{count}/{len(failures)}] "
            f"{label} | {split} | {source_path.name}"
        )

        try:
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Video does not exist: {source_path}"
                )

            sequence, total_frames, fps, selected_indices = (
                preprocess_video(source_path)
            )

            expected_shape = (
                TARGET_FRAMES, IMG_SIZE, IMG_SIZE, 3
            )

            if sequence.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape: {sequence.shape}"
                )

            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            np.save(output_path, sequence)

            # Verify the saved array.
            saved = np.load(output_path, allow_pickle=False)

            if saved.shape != expected_shape:
                raise RuntimeError(
                    f"Saved shape is incorrect: {saved.shape}"
                )

            if saved.dtype != np.uint8:
                raise RuntimeError(
                    f"Saved dtype is incorrect: {saved.dtype}"
                )

            record = {
                "csv_row_index": csv_row_index,
                "source_path": str(source_path),
                "label": label,
                "split": split,
                "output_path": str(
                    output_path.relative_to(OUTPUT_DIR)
                ),
                "num_frames": TARGET_FRAMES,
                "height": IMG_SIZE,
                "width": IMG_SIZE,
                "channels": 3,
                "dtype": str(saved.dtype),
                "original_total_frames": total_frames,
                "fps": fps,
                "selected_frame_indices": json.dumps(
                    selected_indices
                ),
                "status": "success",
            }

            recovered_records.append(record)
            print(
                f"RECOVERED | shape={saved.shape} "
                f"| dtype={saved.dtype}"
            )

        except Exception as error:
            remaining_failures.append({
                "csv_row_index": csv_row_index,
                "source_path": str(source_path),
                "label": label,
                "split": split,
                "error": str(error),
            })
            print("STILL FAILED:", error)

    # Append recovered records without duplicating existing metadata.
    if recovered_records:
        recovered_df = pd.DataFrame(recovered_records)

        existing_ids = set(
            metadata["csv_row_index"].astype(int)
        )

        recovered_df = recovered_df[
            ~recovered_df["csv_row_index"].astype(int).isin(
                existing_ids
            )
        ]

        metadata = pd.concat(
            [metadata, recovered_df],
            ignore_index=True,
        )

        metadata = metadata.sort_values(
            "csv_row_index"
        ).reset_index(drop=True)

        metadata.to_csv(METADATA_PATH, index=False)

    # Replace the failure log with only the videos still failing.
    failure_columns = [
        "csv_row_index",
        "source_path",
        "label",
        "split",
        "error",
    ]

    remaining_df = pd.DataFrame(
        remaining_failures,
        columns=failure_columns,
    )
    remaining_df.to_csv(FAILED_PATH, index=False)

    # Refresh summary from the original split CSV and current metadata.
    original_df = pd.read_csv(CSV_PATH)
    original_df["split"] = (
        original_df["split"].astype(str).str.strip().str.lower()
    )

    metadata = pd.read_csv(METADATA_PATH)
    remaining_df = pd.read_csv(FAILED_PATH)

    split_counts = {}

    for split in ("train", "val", "test"):
        split_counts[split] = {
            "input": int((original_df["split"] == split).sum()),
            "success": int((metadata["split"] == split).sum()),
            "failed": int((remaining_df["split"] == split).sum()),
        }

    summary = {
        "input_csv": str(CSV_PATH),
        "output_directory": str(OUTPUT_DIR),
        "method": "A - Temporal Coverage",
        "target_frames": TARGET_FRAMES,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "storage_dtype": "uint8",
        "input_videos": int(len(original_df)),
        "successful_videos": int(len(metadata)),
        "failed_videos": int(len(remaining_df)),
        "split_counts": split_counts,
        "expected_sequence_shape": [
            TARGET_FRAMES, IMG_SIZE, IMG_SIZE, 3
        ],
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("RECOVERY FINISHED")
    print("=" * 65)
    print("Recovered this run:", len(recovered_records))
    print("Total metadata records:", len(metadata))
    print("Videos still failing:", len(remaining_df))
    print("Updated metadata:", METADATA_PATH)
    print("Updated failure log:", FAILED_PATH)
    print("Updated summary:", SUMMARY_PATH)

    print("\nSplit summary:")
    for split, counts in split_counts.items():
        print(
            f"{split}: input={counts['input']}, "
            f"success={counts['success']}, "
            f"failed={counts['failed']}"
        )


if __name__ == "__main__":
    main()
