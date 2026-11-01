
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
META = PROJECT_DIR / "outputs/preprocessed/metadata.csv"
OUT = PROJECT_DIR / "outputs/augmented/pilot"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(123)
np.random.seed(123)

df = pd.read_csv(META)
train = df[
    (df["split"] == "train") & (df["status"] == "success")
]

# Choose one class for the pilot.
label = "Absent"
samples = train[train["label"] == label].head(3)

if len(samples) == 0:
    raise ValueError(f"No training sequences found for {label}")

for row_number, (_, row) in enumerate(samples.iterrows(), start=1):
    source = Path(row["output_path"])
    if not source.is_absolute():
        source = PROJECT_DIR / "outputs/preprocessed" / source

    sequence = np.load(source)
    if sequence.shape != (12, 224, 224, 3):
        raise ValueError(f"Unexpected shape: {sequence.shape}")

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

    frames = []
    for frame in sequence:
        transformed = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed = np.clip(
            transformed.astype(np.float32) * contrast + brightness,
            0,
            255,
        ).astype(np.uint8)
        frames.append(transformed)

    augmented = np.stack(frames)
    output = OUT / f"{source.stem}_pilot{row_number}.npy"

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    np.save(output, augmented)

    saved = np.load(output)
    print(
        f"Saved: {output.name} | "
        f"shape={saved.shape} | dtype={saved.dtype} | "
        f"size={output.stat().st_size / (1024**2):.2f} MB"
    )

print("\nPilot complete. Original and existing augmented files were not modified.")
