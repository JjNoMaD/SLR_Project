
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA = (
    PROJECT_DIR / "outputs" / "augmented"
    / "augmentation_metadata.csv"
)
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "augmented"
    / "writer_test"
)

# Read-only inspection of existing augmentation metadata.
df = pd.read_csv(METADATA)

if df["augmented_sequence"].duplicated().any():
    raise ValueError("Duplicate paths exist in the current metadata.")

# Use one existing, valid sequence to test reading and copying.
source = Path(str(df.iloc[0]["augmented_sequence"]))

if not source.exists():
    raise FileNotFoundError(source)

sequence = np.load(source, allow_pickle=False)

if sequence.shape != (12, 224, 224, 3) or sequence.dtype != np.uint8:
    raise ValueError("The source sequence has an unexpected format.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
test_output = OUTPUT_DIR / "writer_test_copy.npy"

# Never overwrite a previous test output.
if test_output.exists():
    raise FileExistsError(
        f"{test_output} already exists. Inspect it before rerunning."
    )

np.save(test_output, sequence)

saved = np.load(test_output, allow_pickle=False)

if not np.array_equal(sequence, saved):
    raise ValueError("Writer test failed: saved data differs from source.")

print("Writer test passed.")
print("Source shape:", sequence.shape)
print("Saved shape:", saved.shape)
print("Saved dtype:", saved.dtype)
print("Existing metadata records:", len(df))
print("Existing metadata was not modified.")
print("Test file:", test_output)
