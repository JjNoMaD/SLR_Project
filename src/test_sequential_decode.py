
import cv2
import numpy as np
from pathlib import Path

video_path = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "Absent"
    / "VID20260902002627-converted.mp4"
)

cap = cv2.VideoCapture(str(video_path))

print("Video:", video_path.name)
print("Opened:", cap.isOpened())

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Reported frames:", total)

targets = set(
    np.linspace(
        0,
        max(total - 1, 0),
        min(total, 60),
        dtype=int,
    ).tolist()
)

decoded = 0
target_positions = []

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if decoded in targets:
        target_positions.append(decoded)

    decoded += 1

cap.release()

print("Sequentially decoded:", decoded)
print("Target positions reached:", len(target_positions), "of", len(targets))
print("First target positions:", target_positions[:10])
