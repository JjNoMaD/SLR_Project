
from pathlib import Path
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_PATH = PROJECT_DIR / "sign_language_split.csv"
OUTPUT_PATH = PROJECT_DIR / "sign_language_split_local.csv"

def normalize(name):
    # Ignore spaces and capitalization when matching filenames.
    return re.sub(r"\s+", "", name).casefold()

df = pd.read_csv(CSV_PATH)

extensions = {".mp4", ".mov", ".avi", ".mkv"}
video_lookup = {}

for video in DATASET_DIR.rglob("*"):
    if video.is_file() and video.suffix.lower() in extensions:
        key = (video.parent.name.casefold(), normalize(video.name))
        video_lookup.setdefault(key, []).append(video.resolve())

new_paths = []
missing = []

for _, row in df.iterrows():
    parts = str(row["path"]).replace("\\", "/").rstrip("/").split("/")

    if len(parts) < 2:
        missing.append(str(row["path"]))
        new_paths.append(None)
        continue

    key = (parts[-2].casefold(), normalize(parts[-1]))
    matches = video_lookup.get(key, [])

    if len(matches) == 1:
        new_paths.append(str(matches[0]))
    else:
        missing.append(f"{row['path']} (matches: {len(matches)})")
        new_paths.append(None)

if missing:
    print("Unmatched paths:", len(missing))
    for item in missing[:30]:
        print(item)
    raise SystemExit("No output written. Resolve unmatched paths first.")

df["path"] = new_paths
df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Total rows:", len(df))
print("\nSplit counts:")
print(df["split"].value_counts())
print("Classes:", df["label"].nunique())
print("All paths exist:", all(Path(p).is_file() for p in df["path"]))
