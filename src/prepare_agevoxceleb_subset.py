from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "agevoxceleb"
)

OUTPUT_DIR = METADATA_DIR / "subset"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_FILE = METADATA_DIR / "agevoxceleb_all_labels.csv"


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "00-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+"
]

SAMPLES_PER_CLASS = 2000


# ============================================================
# AGE GROUP
# ============================================================

def age_group(age):

    if age <= 19:
        return "00-19"

    elif age <= 29:
        return "20-29"

    elif age <= 39:
        return "30-39"

    elif age <= 49:
        return "40-49"

    elif age <= 59:
        return "50-59"

    else:
        return "60+"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PREPARING AGEVOXCELEB AGE SUBSET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nTotal labels:", len(df))


# ============================================================
# CREATE 6 AGE GROUPS
# ============================================================

df["age_class"] = df["age"].apply(age_group)


# ============================================================
# SHOW DISTRIBUTION
# ============================================================

print("\nFull distribution:")

print(
    df["age_class"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)


# ============================================================
# TRAIN SUBSET
# ============================================================

train_df = df[df["split"] == "train"].copy()

selected_train = []

print("\nSelecting training samples:")

for class_name in CLASSES:

    class_df = train_df[
        train_df["age_class"] == class_name
    ].copy()

    # Fixed random seed for reproducibility
    class_df = class_df.sample(
        n=min(SAMPLES_PER_CLASS, len(class_df)),
        random_state=42
    )

    selected_train.append(class_df)

    print(
        f"{class_name:>6}: "
        f"{len(class_df)} samples"
    )


train_subset = pd.concat(
    selected_train,
    ignore_index=True
)


# ============================================================
# TEST SET
# ============================================================

test_df = df[df["split"] == "test"].copy()

print("\nTest distribution:")

print(
    test_df["age_class"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)


# ============================================================
# SAVE
# ============================================================

train_output = OUTPUT_DIR / "age_train_subset.csv"
test_output = OUTPUT_DIR / "age_test.csv"

train_subset.to_csv(
    train_output,
    index=False
)

test_df.to_csv(
    test_output,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    "\nTraining:",
    len(train_subset)
)

print(
    "Testing:",
    len(test_df)
)

print("\nTraining classes:")

print(
    train_subset["age_class"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

print("\nSaved:")

print(train_output)
print(test_output)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)