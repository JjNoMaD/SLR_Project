
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent.parent
original_path = (
    project / "outputs/preprocessed/sequences/train/row_000000.npy"
)
augmented_path = (
    project / "outputs/augmented/pilot/row_000000_pilot1.npy"
)

original = np.load(original_path)
augmented = np.load(augmented_path)

# Compare the same frames side by side.
indices = [0, 4, 8, 11]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for col, idx in enumerate(indices):
    axes[0, col].imshow(original[idx])
    axes[0, col].set_title(f"Original — frame {idx}")
    axes[0, col].axis("off")

    axes[1, col].imshow(augmented[idx])
    axes[1, col].set_title(f"Augmented — frame {idx}")
    axes[1, col].axis("off")

plt.tight_layout()
plt.show()
