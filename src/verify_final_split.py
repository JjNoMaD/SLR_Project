
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
csv_path = project_dir / "sign_language_split_final.csv"

df = pd.read_csv(csv_path)

print("Split counts:")
print(df["split"].value_counts())

print("\nVideos per class in each split:")
per_class = pd.crosstab(df["label"], df["split"])
print(per_class.to_string())

print("\nMinimum videos per class:")
print(per_class.min())

print("\nMaximum videos per class:")
print(per_class.max())

print("\nDuplicate paths:", df["path"].str.casefold().duplicated().sum())
print("Paths appearing in multiple splits:",
      (df.groupby(df["path"].str.casefold())["split"].nunique() > 1).sum())
