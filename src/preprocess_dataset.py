
from pathlib import Path
import argparse
import json
import sys
import traceback

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_DIR / "sign_language_split_final.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "preprocessed"

TARGET_FRAMES = 12
MAX_CANDIDATES = 60
IMG_SIZE = 224

REQUIRED_COLUMNS = {"path", "label", "split"}
VALID_SPLITS = {"train", "val", "test"}


# ============================================================
# 2. VIDEO PATH HANDLING
# ============================================================

def resolve_video_path(path_value):
    """Resolve an absolute or project-relative video path."""

    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


# ============================================================
# 3. LOAD CANDIDATE FRAMES
# ============================================================


def load_candidate_frames(video_path):
    """
    Sample candidate frames by decoding the video sequentially.
    Avoid repeated CAP_PROP_POS_FRAMES seeking.

    Returns:
        frames: RGB uint8 array, (N, 224, 224, 3)
        actual_indices: decoded original frame indices
        total_frames: reported total frame count
        fps: video frame rate
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            raise RuntimeError("Video has an invalid frame count.")

        target_indices = np.unique(
            np.linspace(
                0,
                total_frames - 1,
                min(total_frames, MAX_CANDIDATES),
                dtype=int,
            )
        )

        target_set = set(target_indices.tolist())
        frames = []
        actual_indices = []
        frame_idx = 0

        # Decode sequentially instead of seeking to each frame.
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_idx in target_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame,
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )

                frames.append(frame)
                actual_indices.append(frame_idx)

            frame_idx += 1

        if not frames:
            raise RuntimeError(
                "No candidate frames could be decoded sequentially."
            )

        return (
            np.asarray(frames, dtype=np.uint8),
            np.asarray(actual_indices, dtype=np.int64),
            total_frames,
            fps if np.isfinite(fps) and fps > 0 else 0.0,
        )

    finally:
        cap.release()



# ============================================================
# 4. METHOD A: TEMPORAL COVERAGE
# ============================================================

def select_temporal_coverage(
    candidate_indices,
    total_frames,
    target_frames=TARGET_FRAMES,
):
    """
    Select one candidate nearest each temporal-bin midpoint.

    If a bin has no candidate, use the nearest unused candidate.
    If the video has fewer unique candidate frames than the target,
    repeat selected frames to preserve a fixed sequence length.
    """

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    if len(candidate_indices) == 0:
        raise ValueError("No candidate frames available.")

    if total_frames <= 0:
        raise ValueError("Invalid total frame count.")

    boundaries = np.linspace(
        0, total_frames, target_frames + 1
    )

    selected_positions = []

    for segment in range(target_frames):
        start = boundaries[segment]
        end = boundaries[segment + 1]
        midpoint = (start + end) / 2

        candidates = np.where(
            (candidate_indices >= start)
            & (candidate_indices < end)
        )[0]

        if len(candidates) > 0:
            distances = np.abs(
                candidate_indices[candidates] - midpoint
            )
            chosen = int(candidates[np.argmin(distances)])
        else:
            # Fallback: choose the nearest unused candidate.
            available = [
                i for i in range(len(candidate_indices))
                if i not in selected_positions
            ]

            if available:
                chosen = min(
                    available,
                    key=lambda i: abs(
                        candidate_indices[i] - midpoint
                    ),
                )
            else:
                # Extremely short video: repetition is necessary
                # to preserve the fixed sequence length.
                chosen = int(
                    np.argmin(
                        np.abs(candidate_indices - midpoint)
                    )
                )

        selected_positions.append(chosen)

    # Return in chronological order, retaining repeated positions
    # if a very short video requires them.
    selected_positions.sort(
        key=lambda i: candidate_indices[i]
    )

    return selected_positions


# ============================================================
# 5. PREPROCESS ONE VIDEO
# ============================================================

def preprocess_video(video_path):
    """
    Process one video and return a fixed-size frame sequence
    plus information needed for metadata.
    """

    frames, candidate_indices, total_frames, fps = (
        load_candidate_frames(video_path)
    )

    selected_positions = select_temporal_coverage(
        candidate_indices=candidate_indices,
        total_frames=total_frames,
        target_frames=TARGET_FRAMES,
    )

    sequence = frames[selected_positions]

    expected_shape = (
        TARGET_FRAMES,
        IMG_SIZE,
        IMG_SIZE,
        3,
    )

    if sequence.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected sequence shape: {sequence.shape}; "
            f"expected {expected_shape}"
        )

    if sequence.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected dtype: {sequence.dtype}"
        )

    selected_original_indices = [
        int(candidate_indices[i])
        for i in selected_positions
    ]

    return (
        sequence,
        total_frames,
        fps,
        selected_original_indices,
    )


# ============================================================
# 6. MAIN PREPROCESSING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SLR videos using temporal coverage."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N CSV rows for a test run.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Defaults to outputs/preprocessed.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    # Read the original split CSV without modifying it.
    df = pd.read_csv(CSV_PATH)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    # Validate labels and split names.
    if df["label"].isna().any():
        raise ValueError("CSV contains missing labels.")

    df["split"] = df["split"].astype(str).str.strip().str.lower()

    invalid_splits = sorted(
        set(df["split"]) - VALID_SPLITS
    )

    if invalid_splits:
        raise ValueError(
            f"Unexpected split values: {invalid_splits}"
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        df = df.head(args.limit).copy()

    # Keep split-specific output folders.
    for split in VALID_SPLITS:
        (output_dir / "sequences" / split).mkdir(
            parents=True, exist_ok=True
        )

    successful_records = []
    failed_records = []

    total = len(df)

    print("=" * 65)
    print("SLR DATASET PREPROCESSING")
    print("=" * 65)
    print("Input CSV:", CSV_PATH)
    print("Output directory:", output_dir)
    print("Method: A - Temporal Coverage")
    print("Target frames per video:", TARGET_FRAMES)
    print("Image size:", f"{IMG_SIZE} x {IMG_SIZE}")
    print("Videos to process:", total)
    print("=" * 65)

    for count, (row_index, row) in enumerate(
        df.iterrows(), start=1
    ):
        label = str(row["label"]).strip()
        split = str(row["split"]).strip().lower()
        video_path = resolve_video_path(row["path"])

        # Stable output name based on original CSV row position.
        # This avoids filename collisions across classes.
        output_name = f"row_{row_index:06d}.npy"
        output_path = (
            output_dir / "sequences" / split / output_name
        )

        print(
            f"\n[{count}/{total}] "
            f"{label} | {split} | {video_path.name}"
        )

        try:
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file does not exist: {video_path}"
                )

            sequence, total_frames, fps, selected_indices = (
                preprocess_video(video_path)
            )

            # Save as uint8 to reduce storage requirements.
            np.save(output_path, sequence)

            # Verify the saved file can be loaded.
            saved_sequence = np.load(
                output_path, allow_pickle=False
            )

            expected_shape = (
                TARGET_FRAMES,
                IMG_SIZE,
                IMG_SIZE,
                3,
            )

            if saved_sequence.shape != expected_shape:
                raise RuntimeError(
                    f"Saved shape mismatch: {saved_sequence.shape}"
                )

            if saved_sequence.dtype != np.uint8:
                raise RuntimeError(
                    f"Saved dtype mismatch: {saved_sequence.dtype}"
                )

            successful_records.append({
                "csv_row_index": int(row_index),
                "source_path": str(video_path),
                "label": label,
                "split": split,
                "output_path": str(
                    output_path.relative_to(output_dir)
                ),
                "num_frames": int(TARGET_FRAMES),
                "height": int(IMG_SIZE),
                "width": int(IMG_SIZE),
                "channels": 3,
                "dtype": str(saved_sequence.dtype),
                "original_total_frames": int(total_frames),
                "fps": float(fps),
                "selected_frame_indices": json.dumps(
                    selected_indices
                ),
                "status": "success",
            })

            print(
                f"SUCCESS | shape={saved_sequence.shape} "
                f"| dtype={saved_sequence.dtype}"
            )

        except Exception as error:
            failed_records.append({
                "csv_row_index": int(row_index),
                "source_path": str(video_path),
                "label": label,
                "split": split,
                "error": str(error),
            })

            print("FAILED:", error)

    # Save metadata and failure logs.
    metadata_path = output_dir / "metadata.csv"
    failures_path = output_dir / "failed_videos.csv"
    summary_path = output_dir / "preprocessing_summary.json"

    metadata_columns = [
        "csv_row_index",
        "source_path",
        "label",
        "split",
        "output_path",
        "num_frames",
        "height",
        "width",
        "channels",
        "dtype",
        "original_total_frames",
        "fps",
        "selected_frame_indices",
        "status",
    ]

    failures_columns = [
        "csv_row_index",
        "source_path",
        "label",
        "split",
        "error",
    ]

    metadata_df = pd.DataFrame(
        successful_records,
        columns=metadata_columns,
    )

    failures_df = pd.DataFrame(
        failed_records,
        columns=failures_columns,
    )

    metadata_df.to_csv(metadata_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    # Summary for this run.
    split_counts = {}

    for split in sorted(VALID_SPLITS):
        split_counts[split] = {
            "input": int((df["split"] == split).sum()),
            "success": int(
                (metadata_df["split"] == split).sum()
            ) if not metadata_df.empty else 0,
            "failed": int(
                (failures_df["split"] == split).sum()
            ) if not failures_df.empty else 0,
        }

    summary = {
        "input_csv": str(CSV_PATH),
        "output_directory": str(output_dir),
        "method": "A - Temporal Coverage",
        "target_frames": TARGET_FRAMES,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "storage_dtype": "uint8",
        "input_videos": int(total),
        "successful_videos": int(len(successful_records)),
        "failed_videos": int(len(failed_records)),
        "split_counts": split_counts,
        "expected_sequence_shape": [
            TARGET_FRAMES, IMG_SIZE, IMG_SIZE, 3
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("PREPROCESSING FINISHED")
    print("=" * 65)
    print("Input videos:", total)
    print("Successful:", len(successful_records))
    print("Failed:", len(failed_records))
    print("Metadata:", metadata_path)
    print("Failure log:", failures_path)
    print("Summary:", summary_path)

    print("\nSplit summary:")
    for split, counts in split_counts.items():
        print(
            f"{split}: input={counts['input']}, "
            f"success={counts['success']}, "
            f"failed={counts['failed']}"
        )

    if failed_records:
        print(
            "\nSome videos failed. Check failed_videos.csv "
            "before training."
        )

    if len(successful_records) != total:
        print(
            "\nWARNING: Not every input video was processed "
            "successfully."
        )

    print("\nNo changes were made to the source CSV.")


if __name__ == "__main__":
    main()
