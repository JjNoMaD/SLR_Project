
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_final.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "preprocessed"

EXPECTED_SHAPE = (12, 224, 224, 3)
EXPECTED_SPLITS = {"train": 700, "val": 150, "test": 200}


def main():
    original = pd.read_csv(CSV_PATH)
    metadata_path = OUTPUT_DIR / "metadata.csv"
    failure_path = OUTPUT_DIR / "failed_videos.csv"

    metadata = pd.read_csv(metadata_path)
    failures = pd.read_csv(failure_path)

    problems = []

    print("=" * 60)
    print("PREPROCESSED DATASET INTEGRITY CHECK")
    print("=" * 60)

    # 1. Verify row counts and unique source records.
    print("Original CSV rows:", len(original))
    print("Metadata rows:", len(metadata))
    print("Failure rows:", len(failures))

    if len(original) != 1050:
        problems.append("Original CSV does not contain 1050 rows.")

    if len(metadata) != len(original):
        problems.append("Metadata count differs from original CSV.")

    if metadata["csv_row_index"].duplicated().any():
        problems.append("Duplicate CSV row indices in metadata.")

    if len(failures) != 0:
        problems.append("Failure log is not empty.")

    # 2. Check split counts in the original CSV and metadata.
    original_splits = (
        original["split"].astype(str).str.strip().str.lower()
    )
    metadata_splits = (
        metadata["split"].astype(str).str.strip().str.lower()
    )

    print("\nSplit counts:")
    for split, expected in EXPECTED_SPLITS.items():
        source_count = int((original_splits == split).sum())
        output_count = int((metadata_splits == split).sum())

        print(
            f"{split}: source={source_count}, "
            f"metadata={output_count}, expected={expected}"
        )

        if source_count != expected or output_count != expected:
            problems.append(f"Incorrect count for split: {split}")

    # 3. Verify every metadata entry against the source CSV.
    source_by_row = original.reset_index().set_index("index")

    for _, record in metadata.iterrows():
        row_id = int(record["csv_row_index"])

        if row_id not in source_by_row.index:
            problems.append(f"Unknown source CSV row: {row_id}")
            continue

        source = source_by_row.loc[row_id]

        if str(source["label"]).strip() != str(record["label"]).strip():
            problems.append(f"Label mismatch at CSV row {row_id}")

        if (
            str(source["split"]).strip().lower()
            != str(record["split"]).strip().lower()
        ):
            problems.append(f"Split mismatch at CSV row {row_id}")

        expected_source = Path(str(source["path"]).strip())
        recorded_source = Path(str(record["source_path"]).strip())

        if (
            not expected_source.is_absolute()
            and not recorded_source.is_absolute()
        ):
            expected_source = PROJECT_DIR / expected_source
            recorded_source = PROJECT_DIR / recorded_source

        if str(expected_source.resolve()).casefold() != (
            str(recorded_source.resolve()).casefold()
        ):
            problems.append(f"Source path mismatch at CSV row {row_id}")

    # 4. Verify every saved NumPy array.
    checked = 0
    seen_outputs = set()

    print("\nChecking saved arrays...")

    for _, record in metadata.iterrows():
        output_path = OUTPUT_DIR / str(record["output_path"])

        resolved_output = str(output_path.resolve()).casefold()
        if resolved_output in seen_outputs:
            problems.append(f"Duplicate output path: {output_path}")
            continue

        seen_outputs.add(resolved_output)

        if not output_path.is_file():
            problems.append(f"Missing array: {output_path}")
            continue

        try:
            array = np.load(output_path, mmap_mode="r", allow_pickle=False)

            if array.shape != EXPECTED_SHAPE:
                problems.append(
                    f"Wrong shape {array.shape}: {output_path}"
                )

            if array.dtype != np.uint8:
                problems.append(
                    f"Wrong dtype {array.dtype}: {output_path}"
                )

            if not np.isfinite(array).all():
                problems.append(f"Non-finite pixel values: {output_path}")

            checked += 1

        except Exception as error:
            problems.append(
                f"Cannot read {output_path}: {error}"
            )

    print("Arrays checked:", checked)

    # 5. Verify total output count.
    all_arrays = list(
        (OUTPUT_DIR / "sequences").rglob("*.npy")
    )

    print("Total .npy files in sequences:", len(all_arrays))

    if len(all_arrays) != len(metadata):
        problems.append(
            "Number of .npy files differs from metadata count."
        )

    # 6. Report result.
    print("\n" + "=" * 60)

    if problems:
        print(f"VERIFICATION FAILED: {len(problems)} issue(s)")
        for problem in problems[:50]:
            print("-", problem)

        if len(problems) > 50:
            print("-", len(problems) - 50, "additional issues")

        sys.exit(1)

    print("VERIFICATION PASSED")
    print("All 1050 source records have matching metadata.")
    print("All saved arrays have the expected shape and dtype.")
    print("All train/validation/test counts match.")
    print("No failed videos or duplicate output paths.")
    print("=" * 60)


if __name__ == "__main__":
    main()
