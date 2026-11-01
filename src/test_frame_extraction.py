
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_final.csv"

TARGET_FRAMES = 12
MAX_CANDIDATES = 60
IMG_SIZE = 224
FEATURE_SIZE = 64


# ============================================================
# 2. LOAD DATASET SPLIT
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Split CSV not found: {CSV_PATH}\n"
        "Check that sign_language_split_final.csv is in SLR_Project."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {"path", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {sorted(required_columns)}"
    )

# Select one training video for testing.
train_df = df[df["split"] == "train"]

if train_df.empty:
    raise ValueError("No training videos found in the CSV.")

row = train_df.iloc[0]
video_path = Path(row["path"])

if not video_path.is_file():
    raise FileNotFoundError(f"Video not found: {video_path}")


# ============================================================
# 3. TEMPORAL COVERAGE + VISUAL DIVERSITY FRAME SELECTION
# ============================================================

def select_diverse_frames(
    video_path,
    target_frames=TARGET_FRAMES,
    max_candidates=MAX_CANDIDATES,
):
    """
    Select a fixed number of frames distributed throughout a video.

    Method:
    1. Sample candidate frames across the timeline.
    2. Convert candidates to grayscale visual descriptors.
    3. Divide the timeline into temporal segments.
    4. Select a visually distinctive frame in each segment.

    Returns:
        selected_frames: RGB frames resized to 224 x 224.
        selected_indices: Original video frame indices.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")

    candidate_indices = np.unique(
        np.linspace(
            0,
            total_frames - 1,
            min(total_frames, max_candidates),
        ).astype(int)
    )

    frames = []
    actual_indices = []
    features = []

    for frame_idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        if not success:
            continue

        # Convert BGR to RGB and resize for model input.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # Compact grayscale descriptor for diversity comparison.
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(
            gray,
            (FEATURE_SIZE, FEATURE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        descriptor = gray.astype(np.float32) / 255.0

        frames.append(frame_rgb)
        actual_indices.append(int(frame_idx))
        features.append(descriptor.flatten())

    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames in: {video_path}")

    features = np.asarray(features, dtype=np.float32)

    n_candidates = len(frames)
    n_selected = min(target_frames, n_candidates)

    selected_positions = []

    # Divide candidate positions into temporal segments.
    # One frame is selected from every segment to preserve coverage.
    for segment in range(n_selected):
        start = round(segment * n_candidates / n_selected)
        end = round((segment + 1) * n_candidates / n_selected)

        end = max(end, start + 1)
        candidates = list(range(start, min(end, n_candidates)))

        if not selected_positions:
            # Start with a frame near the center of the first segment.
            chosen = candidates[len(candidates) // 2]
        else:
            # Select the frame most visually different from those
            # already selected, using mean absolute pixel difference.
            scores = []

            for candidate in candidates:
                distances = [
                    np.mean(
                        np.abs(features[candidate] - features[selected])
                    )
                    for selected in selected_positions
                ]

                scores.append(float(min(distances)))

            chosen = candidates[int(np.argmax(scores))]

        selected_positions.append(chosen)

    selected_positions.sort()

    selected_frames = [
        frames[position] for position in selected_positions
    ]

    selected_indices = [
        actual_indices[position] for position in selected_positions
    ]

    return selected_frames, selected_indices


# ============================================================
# 4. EXTRACT FRAMES
# ============================================================

selected_frames, selected_indices = select_diverse_frames(video_path)

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

duration = total_frames / fps if fps > 0 else 0.0

print("=" * 55)
print("FRAME EXTRACTION RESULTS")
print("=" * 55)
print("Class:", row["label"])
print("Video:", video_path.name)
print("Total video frames:", total_frames)
print("FPS:", fps)
print("Duration (seconds):", round(duration, 2))
print("Candidate limit:", MAX_CANDIDATES)
print("Selected frame count:", len(selected_frames))
print("Selected frame indices:", selected_indices)
print("Selected timestamps (seconds):", [
    round(i / fps, 3) if fps > 0 else None
    for i in selected_indices
])
print("Frame shape:", selected_frames[0].shape)


# ============================================================
# 5. VISUALIZE SELECTED FRAMES
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, frame, frame_idx in zip(
    axes, selected_frames, selected_indices
):
    ax.imshow(frame)
    timestamp = frame_idx / fps if fps > 0 else 0
    ax.set_title(
        f"Frame {frame_idx}\n{timestamp:.2f} sec"
    )

plt.suptitle(
    f"Temporal-Coverage Frame Selection: {row['label']}",
    fontsize=15,
)
plt.tight_layout()
plt.show()
