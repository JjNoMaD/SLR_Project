
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_PATH = PROJECT_DIR / "sign_language_split.csv"

df = pd.read_csv(CSV_PATH)

# Show actual files in the Absent class
absent_dir = DATASET_DIR / "Absent"

print("Absent folder exists:", absent_dir.exists())

if absent_dir.exists():
    print("\nFirst 30 actual video filenames:")
    for p in sorted(absent_dir.iterdir())[:30]:
        if p.is_file():
            print(p.name)

print("\nCSV filenames for Absent:")
absent_rows = df[
    df["path"].astype(str).str.replace("\\", "/", regex=False)
      .str.contains("/Absent/", case=False, regex=False)
]

for p in absent_rows["path"].head(20):
    print(Path(str(p)).name)
