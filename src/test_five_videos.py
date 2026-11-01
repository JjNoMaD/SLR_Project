
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_final.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "frame_comparison"

TARGET_FRAMES = 12
MAX_CANDIDATES = 60
IMG_SIZE = 224
FEATURE_SIZE = 64

TEST_LABELS = ["Absent", "Garlic", "Mango", "cabbage", "Book"]


# ============================================================
# 2. LOAD CANDIDATE FRAMES
# ============================================================

def load_candidate_frames(video_path):
    """Sample candidate frames uniformly across a video."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")

    candidate_indices = np.unique(
        np.linspace(
            0,
            total_frames - 1,
            min(total_frames, MAX_CANDIDATES),
            dtype=int,
        )
    )

    frames = []
    actual_indices = []

    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        frames.append(frame)
        actual_indices.append(int(idx))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted: {video_path}")

    return (
        np.asarray(frames, dtype=np.uint8),
        np.asarray(actual_indices, dtype=int),
        total_frames,
        fps if fps > 0 else 0.0,
    )


# ============================================================
# 3. METHOD A: TEMPORAL COVERAGE BASELINE
# ============================================================

def select_temporal_coverage(
    candidate_indices,
    total_frames,
    target_frames=TARGET_FRAMES,
):
    """Select the candidate nearest each temporal-bin midpoint."""

    selected = []
    boundaries = np.linspace(0, total_frames, target_frames + 1)

    for segment in range(target_frames):
        start = boundaries[segment]
        end = boundaries[segment + 1]

        candidates = np.where(
            (candidate_indices >= start)
            & (candidate_indices < end)
        )[0]

        if len(candidates) == 0:
            continue

        midpoint = (start + end) / 2
        chosen = candidates[
            np.argmin(np.abs(candidate_indices[candidates] - midpoint))
        ]

        selected.append(int(chosen))

    return sorted(set(selected))


# ============================================================
# 4. COMMON VISUAL DESCRIPTORS
# ============================================================

def make_descriptors(frames):
    """Create compact grayscale descriptors for comparison."""

    descriptors = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(
            gray,
            (FEATURE_SIZE, FEATURE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        descriptors.append(gray.astype(np.float32) / 255.0)

    return np.asarray(descriptors, dtype=np.float32)


def choose_motion_diverse(
    descriptors,
    candidate_indices,
    total_frames,
    target_frames=TARGET_FRAMES,
):
    """
    Select one frame per temporal bin using motion, diversity,
    and proximity to the temporal-bin midpoint.
    """

    count = len(descriptors)

    if count == 0:
        return []

    motion = np.zeros(count, dtype=np.float32)

    for i in range(1, count):
        motion[i] = np.mean(
            np.abs(descriptors[i] - descriptors[i - 1])
        )

    max_motion = float(motion.max())

    if max_motion > 1e-8:
        motion /= max_motion

    boundaries = np.linspace(0, total_frames, target_frames + 1)
    selected = []

    for segment in range(target_frames):
        start = boundaries[segment]
        end = boundaries[segment + 1]

        candidates = np.where(
            (candidate_indices >= start)
            & (candidate_indices < end)
        )[0]

        if len(candidates) == 0:
            continue

        midpoint = (start + end) / 2
        best_idx = None
        best_score = -np.inf

        for idx in candidates:
            temporal_distance = abs(
                candidate_indices[idx] - midpoint
            ) / max((end - start) / 2, 1.0)

            temporal_score = 1.0 - min(temporal_distance, 1.0)

            if selected:
                distances = [
                    float(np.mean(np.abs(
                        descriptors[idx] - descriptors[previous]
                    )))
                    for previous in selected
                ]
                diversity = min(distances)
                diversity = min(diversity / 0.25, 1.0)
            else:
                diversity = 0.5

            score = (
                0.55 * float(motion[idx])
                + 0.30 * diversity
                + 0.15 * temporal_score
            )

            if score > best_score:
                best_score = score
                best_idx = int(idx)

        if best_idx is not None:
            selected.append(best_idx)

    return sorted(set(selected))


# ============================================================
# 5. METHOD B: WHOLE-FRAME MOTION + DIVERSITY
# ============================================================

def select_motion_diverse(
    frames,
    candidate_indices,
    total_frames,
    target_frames=TARGET_FRAMES,
):
    descriptors = make_descriptors(frames)

    return choose_motion_diverse(
        descriptors,
        candidate_indices,
        total_frames,
        target_frames,
    )


# ============================================================
# 6. METHOD C: UPPER-BODY MOTION + DIVERSITY
# ============================================================

def select_upper_body_focused(
    frames,
    candidate_indices,
    total_frames,
    target_frames=TARGET_FRAMES,
):
    """
    Use an approximate central upper-body crop for selection scoring.

    This is a fixed crop heuristic, not a hand/body detector.
    The original full frames are retained for model input.
    """

    cropped_descriptors = []

    for frame in frames:
        height, width = frame.shape[:2]

        x1 = int(0.20 * width)
        x2 = int(0.85 * width)
        y1 = int(0.15 * height)
        y2 = int(0.90 * height)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            raise RuntimeError("Upper-body crop is empty.")

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(
            gray,
            (FEATURE_SIZE, FEATURE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        cropped_descriptors.append(
            gray.astype(np.float32) / 255.0
        )

    cropped_descriptors = np.asarray(
        cropped_descriptors, dtype=np.float32
    )

    return choose_motion_diverse(
        cropped_descriptors,
        candidate_indices,
        total_frames,
        target_frames,
    )


# ============================================================
# 7. VISUALIZE ALL THREE METHODS
# ============================================================

def save_comparison(
    label,
    video_name,
    frames,
    candidate_indices,
    fps,
    selections,
):
    """
    Rows:
      A = temporal coverage
      B = whole-frame motion + diversity
      C = upper-body motion + diversity
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = [
        ("Method A: Temporal coverage", selections["A"]),
        ("Method B: Whole-frame motion + diversity", selections["B"]),
        ("Method C: Upper-body motion + diversity", selections["C"]),
    ]

    fig, axes = plt.subplots(
        3,
        TARGET_FRAMES,
        figsize=(20, 9),
        squeeze=False,
    )

    for row_idx, (method_name, selected) in enumerate(methods):
        for col_idx in range(TARGET_FRAMES):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(selected):
                continue

            candidate_idx = selected[col_idx]
            frame = frames[candidate_idx]
            original_idx = int(candidate_indices[candidate_idx])

            timestamp = (
                original_idx / fps if fps > 0 else 0.0
            )

            ax.imshow(frame)
            ax.set_title(
                f"Frame {original_idx}\n{timestamp:.2f}s",
                fontsize=8,
            )

        axes[row_idx, 0].set_ylabel(
            method_name,
            fontsize=9,
            rotation=90,
            labelpad=10,
        )

    fig.suptitle(
        f"{label} — {video_name}\n"
        "Keyframe Selection Comparison",
        fontsize=14,
    )

    plt.tight_layout(rect=[0.035, 0, 1, 0.94])

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in f"{label}_{Path(video_name).stem}"
    )

    output_path = OUTPUT_DIR / f"{safe_name}_three_methods.png"

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ============================================================
# 8. MAIN TEST
# ============================================================

def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"path", "label", "split"}

    if not required.issubset(df.columns):
        raise ValueError(f"CSV requires columns: {sorted(required)}")

    # Use training examples only for selector development.
    train_df = df[
        df["split"].astype(str).str.lower() == "train"
    ]

    test_rows = []

    for label in TEST_LABELS:
        matches = train_df[
            train_df["label"].astype(str).str.casefold()
            == label.casefold()
        ]

        if matches.empty:
            print(f"WARNING: No training video found for {label}")
            continue

        test_rows.append(matches.iloc[0])

    if not test_rows:
        raise RuntimeError("No matching videos found.")

    print("=" * 65)
    print("THREE-METHOD KEYFRAME SELECTION COMPARISON")
    print("=" * 65)
    print("Videos selected:", len(test_rows))
    print("Output folder:", OUTPUT_DIR)

    for row in test_rows:
        label = str(row["label"])
        video_path = Path(str(row["path"]))

        print("\n" + "=" * 65)
        print("Class:", label)
        print("Video:", video_path.name)

        if not video_path.is_file():
            print("SKIPPED: File does not exist.")
            continue

        try:
            frames, indices, total_frames, fps = load_candidate_frames(
                video_path
            )

            selections = {
                "A": select_temporal_coverage(indices, total_frames),
                "B": select_motion_diverse(
                    frames, indices, total_frames
                ),
                "C": select_upper_body_focused(
                    frames, indices, total_frames
                ),
            }

            duration = (
                total_frames / fps if fps > 0 else 0.0
            )

            print("Total frames:", total_frames)
            print("FPS:", round(fps, 2))
            print("Duration (seconds):", round(duration, 2))
            print("Candidate frames:", len(frames))

            for method, selected in selections.items():
                original_indices = [
                    int(indices[i]) for i in selected
                ]

                print(
                    f"Method {method}: "
                    f"{len(selected)} frames -> {original_indices}"
                )

            output_path = save_comparison(
                label=label,
                video_name=video_path.name,
                frames=frames,
                candidate_indices=indices,
                fps=fps,
                selections=selections,
            )

            print("Saved comparison:", output_path)

        except Exception as error:
            print("ERROR:", error)

    print("\nAll comparisons finished.")
    print("Open the *_three_methods.png files in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
