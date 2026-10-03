from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "agevoxceleb" / "repository"

TRAIN_FILE = DATA_DIR / "utt2age.train"
TEST_FILE = DATA_DIR / "utt2age.test"


# ============================================================
# AGE GROUP FUNCTION
# ============================================================

def age_group(age):
    if age <= 12:
        return "00-12"
    elif age <= 19:
        return "13-19"
    elif age <= 29:
        return "20-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    elif age <= 79:
        return "70-79"
    else:
        return "80+"


# ============================================================
# LOAD FILE
# ============================================================

def load_age_file(file_path, split_name):

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            utterance_id = parts[0]

            try:
                age = float(parts[1])
            except ValueError:
                continue

            rows.append({
                "utterance_id": utterance_id,
                "age": age,
                "age_group": age_group(age),
                "split": split_name
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("AGEVOXCELEB AGE DATASET ANALYSIS")
print("=" * 70)

print("\nLoading training labels...")

train_df = load_age_file(TRAIN_FILE, "train")

print("Training samples:", len(train_df))

print("\nLoading test labels...")

test_df = load_age_file(TEST_FILE, "test")

print("Test samples:", len(test_df))


# ============================================================
# COMBINE
# ============================================================

df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("Total labelled utterances:", len(df))

print("Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())
print("Mean age:", round(df["age"].mean(), 2))
print("Median age:", df["age"].median())


# ============================================================
# AGE GROUP DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("AGE GROUP DISTRIBUTION")
print("=" * 70)

group_order = [
    "00-12",
    "13-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80+"
]

group_counts = (
    df["age_group"]
    .value_counts()
    .reindex(group_order, fill_value=0)
)

for group, count in group_counts.items():

    percentage = count / len(df) * 100

    print(
        f"{group:>6} : "
        f"{count:>8} samples "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SENIOR CITIZEN DISTRIBUTION
# ============================================================

senior_df = df[df["age"] >= 60]

print("\n" + "=" * 70)
print("SENIOR CITIZEN DATA")
print("=" * 70)

print("60+ samples:", len(senior_df))

if len(senior_df) > 0:

    print(
        "Percentage:",
        round(len(senior_df) / len(df) * 100, 2),
        "%"
    )

    print(
        "Minimum senior age:",
        senior_df["age"].min()
    )

    print(
        "Maximum senior age:",
        senior_df["age"].max()
    )


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST DISTRIBUTION")
print("=" * 70)

for split in ["train", "test"]:

    subset = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("Samples:", len(subset))

    counts = (
        subset["age_group"]
        .value_counts()
        .reindex(group_order, fill_value=0)
    )

    for group, count in counts.items():

        print(
            f"  {group:>6} : {count:>8}"
        )


# ============================================================
# CHECK UTTERANCE ID FORMAT
# ============================================================

print("\n" + "=" * 70)
print("UTTERANCE ID EXAMPLES")
print("=" * 70)

print(df.head(10).to_string(index=False))


# ============================================================
# UNIQUE SPEAKER IDS
# ============================================================

def extract_speaker_id(utterance_id):

    # AgeVoxCeleb/VoxCeleb style IDs normally begin
    # with the speaker identifier.

    parts = utterance_id.split("/")

    if len(parts) > 0:
        return parts[0]

    return utterance_id


df["speaker_id"] = df["utterance_id"].apply(
    extract_speaker_id
)

print("\n" + "=" * 70)
print("SPEAKER INFORMATION")
print("=" * 70)

print(
    "Unique speakers:",
    df["speaker_id"].nunique()
)


# ============================================================
# SENIOR SPEAKER INFORMATION
# ============================================================

speaker_age = (
    df.groupby("speaker_id")["age"]
    .agg(["min", "max", "mean", "count"])
    .reset_index()
)

senior_speakers = speaker_age[
    speaker_age["mean"] >= 60
]

print(
    "Speakers with mean age >= 60:",
    len(senior_speakers)
)


# ============================================================
# SAVE ANALYSIS CSV
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata" / "agevoxceleb"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = OUTPUT_DIR / "agevoxceleb_all_labels.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nSaved metadata:")
print(output_file)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)