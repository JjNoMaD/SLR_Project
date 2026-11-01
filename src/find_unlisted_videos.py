
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_PATH = PROJECT_DIR / "sign_language_split_clean.csv"

df = pd.read_csv(CSV_PATH)

# Get all videos physically present in the dataset.
extensions = {".mp4", ".mov", ".avi", ".mkv"}
disk_videos = {
    str(p.resolve()).casefold(): p.resolve()
    for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
}

# Get videos referenced by the cleaned CSV.
csv_paths = {
    str((PROJECT_DIR / p).resolve() if not Path(p).is_absolute() else Path(p))
    .casefold()
    for p in df["path"]
}

unlisted = [
    path for key, path in disk_videos.items()
    if key not in csv_paths
]

print("Unique video files on disk:", len(disk_videos))
print("Unique paths in cleaned CSV:", df["path"].nunique())
print("Videos not listed in CSV:", len(unlisted))

for path in sorted(unlisted):
    print(path)
