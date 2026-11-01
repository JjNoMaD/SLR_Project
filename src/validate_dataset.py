
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_local.csv"

df = pd.read_csv(CSV_PATH)

print("Total CSV rows:", len(df))
print("\nSplit counts:")
print(df["split"].value_counts())

print("\nNumber of classes:", df["label"].nunique())
print("Classes per split:")
print(df.groupby("split")["label"].nunique())

missing = [
    p
    for p in df["path"]
    if not (
        Path(p) if Path(p).is_absolute() else PROJECT_DIR / p
    ).is_file()
]
print("\nMissing video paths:", len(missing))

if missing:
    print("First missing paths:")
    for path in missing[:10]:
        print(path)

# Check duplicate paths in the CSV
print("Duplicate paths:", df["path"].duplicated().sum())

# Check whether a video appears in more than one split
split_counts = df.groupby("path")["split"].nunique()
print("Paths appearing in multiple splits:", (split_counts > 1).sum())

if len(df) == 1050 and not missing:
    print("\nBasic dataset validation passed.")
else:
    print("\nDataset validation needs attention.")
