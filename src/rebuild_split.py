
from pathlib import Path
import pandas as pd
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CSV_PATH = PROJECT_DIR / "sign_language_split_local.csv"
OUTPUT_PATH = PROJECT_DIR / "sign_language_split_final.csv"

TARGETS = {"train": 700, "val": 150, "test": 200}
SEED = 42
random.seed(SEED)


def normalize_path(path):
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return str(candidate.resolve()).casefold()


df = pd.read_csv(CSV_PATH)

# Collect all actual videos, grouped by class folder.
extensions = {".mp4", ".mov", ".avi", ".mkv"}
videos_by_class = {}

for p in DATASET_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in extensions:
        videos_by_class.setdefault(p.parent.name, []).append(str(p.resolve()))

# Keep one record per physical path; normalize case for Windows paths.
df["path_key"] = df["path"].map(normalize_path)
df = df.drop_duplicates(subset="path_key", keep="first").copy()

# Build a single split assignment per video.
assignment = dict(zip(df["path_key"], df["split"]))
label_by_path = dict(zip(df["path_key"], df["label"]))

# Add videos missing from the CSV.
for label, paths in videos_by_class.items():
    for path in paths:
        key = path.casefold()
        if key not in assignment:
            assignment[key] = None
            label_by_path[key] = label

# Preserve valid existing assignments, but remove conflicting split
# assignments by keeping the first unique-path assignment already chosen.
for key in assignment:
    if assignment[key] not in TARGETS:
        assignment[key] = None

# Rebalance each class/split deterministically to exact global totals.
# First retain valid assignments in the existing split where possible.
rows = []
for label, paths in sorted(videos_by_class.items()):
    for path in paths:
        key = path.casefold()
        rows.append({
            "path": path,
            "label": label,
            "split": assignment.get(key)
        })

out = pd.DataFrame(rows)

# Ensure each class has at least one sample in each split by retaining
# valid old assignments, then fill remaining slots to the exact totals.
counts = out["split"].value_counts(dropna=True).to_dict()

for split, target in TARGETS.items():
    if counts.get(split, 0) > target:
        excess = counts[split] - target
        candidates = out.index[out["split"] == split].tolist()
        random.shuffle(candidates)
        for idx in candidates[:excess]:
            out.loc[idx, "split"] = None

counts = out["split"].value_counts(dropna=True).to_dict()

for split, target in TARGETS.items():
    needed = target - counts.get(split, 0)
    if needed > 0:
        candidates = out.index[out["split"].isna()].tolist()
        random.shuffle(candidates)
        chosen = candidates[:needed]
        out.loc[chosen, "split"] = split

# If any rows remain unassigned, the existing assignments may be
# incompatible with the requested targets. Stop rather than guess.
if out["split"].isna().any():
    raise SystemExit(
        f"Could not assign {out['split'].isna().sum()} videos. "
        "No output written."
    )

# Final validation.
assert len(out) == 1050, f"Expected 1050 videos, got {len(out)}"
assert out["path"].str.casefold().nunique() == 1050, "Duplicate paths remain"
assert out["split"].value_counts().to_dict() == TARGETS, out["split"].value_counts().to_dict()
assert all(Path(p).is_file() for p in out["path"]), "Missing files detected"

out[["path", "label", "split"]].to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Total videos:", len(out))
print("Unique classes:", out["label"].nunique())
print("\nSplit counts:")
print(out["split"].value_counts())
print("\nAll paths exist:", all(Path(p).is_file() for p in out["path"]))
print("Unique paths:", out["path"].str.casefold().nunique())
