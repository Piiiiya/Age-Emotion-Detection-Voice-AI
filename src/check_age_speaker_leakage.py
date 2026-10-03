from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "agevoxceleb"
    / "subset"
)

TRAIN_FILE = SUBSET_DIR / "age_train_subset.csv"
TEST_FILE = SUBSET_DIR / "age_test.csv"


# ============================================================
# SPEAKER EXTRACTION
# ============================================================

def extract_speaker_id(utterance_id):
    """
    AgeVoxCeleb ID:

    speaker_id/video_id/utterance_id

    Example:
    id00012/aE4Om0EEiuk/00115

    Speaker:
    id00012
    """

    return str(utterance_id).split("/")[0]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGEVOXCELEB SPEAKER LEAKAGE CHECK")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# ============================================================
# EXTRACT SPEAKERS
# ============================================================

train_df["speaker_id"] = train_df["utterance_id"].apply(
    extract_speaker_id
)

test_df["speaker_id"] = test_df["utterance_id"].apply(
    extract_speaker_id
)


train_speakers = set(
    train_df["speaker_id"]
)

test_speakers = set(
    test_df["speaker_id"]
)


# ============================================================
# OVERLAP
# ============================================================

overlap = train_speakers.intersection(
    test_speakers
)


print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print(
    "\nUnique training speakers:",
    len(train_speakers)
)

print(
    "Unique testing speakers:",
    len(test_speakers)
)

print(
    "\nSpeaker overlap:",
    len(overlap)
)


# ============================================================
# RESULT
# ============================================================

if len(overlap) == 0:

    print("\n" + "=" * 70)
    print("PASS")
    print("=" * 70)

    print(
        "\nNo speaker appears in both training and testing."
    )

    print(
        "Speaker leakage: NONE"
    )

else:

    print("\n" + "=" * 70)
    print("WARNING")
    print("=" * 70)

    print(
        "\nSpeakers appearing in both sets:",
        len(overlap)
    )

    print("\nFirst overlapping speakers:")

    for speaker in sorted(overlap)[:20]:
        print(" ", speaker)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    train_df["age_class"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

print(
    test_df["age_class"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)