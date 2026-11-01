
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_local.csv"
OUTPUT_PATH = PROJECT_DIR / "sign_language_split_clean.csv"

df = pd.read_csv(CSV_PATH)

# Remove repeated rows for the same video, keeping the first occurrence.
df = df.drop_duplicates(subset=["path"], keep="first").copy()

# Check that no video is assigned to multiple splits.
assert df["path"].is_unique, "Duplicate video paths remain!"

# Report new split counts. Removing duplicates may reduce the original totals.
print("Rows after removing duplicates:", len(df))
print("\nSplit counts:")
print(df["split"].value_counts())

print("\nClasses per split:")
print(df.groupby("split")["label"].nunique())

# Save separately; never overwrite the original split CSV.
df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH)
print("Note: split counts may be below 700/150/200 after deduplication.")
