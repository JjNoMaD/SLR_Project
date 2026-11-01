
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "sign_language_split_local.csv"

df = pd.read_csv(CSV_PATH)

# Show every repeated path and its split assignments
duplicates = df[df["path"].duplicated(keep=False)].sort_values("path")

print("=== Duplicate CSV paths ===")
if duplicates.empty:
    print("None")
else:
    print(duplicates[["path", "label", "split"]].to_string(index=False))

print("\n=== Same path assigned to multiple splits ===")
split_counts = df.groupby("path")["split"].nunique()
cross_split_paths = split_counts[split_counts > 1].index

cross_split = df[df["path"].isin(cross_split_paths)].sort_values("path")
if cross_split.empty:
    print("None")
else:
    print(cross_split[["path", "label", "split"]].to_string(index=False))
