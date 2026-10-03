from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "agevoxceleb"
    / "subset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "agevoxceleb"
)

TRAIN_FILE = SUBSET_DIR / "age_train_subset.csv"
TEST_FILE = SUBSET_DIR / "age_test.csv"


def extract_speaker_id(utterance_id):
    return str(utterance_id).split("/")[0]


def extract_video_id(utterance_id):
    parts = str(utterance_id).split("/")

    if len(parts) >= 2:
        return parts[1]

    return ""


print("=" * 70)
print("PREPARING AGEVOXCELEB AUDIO MANIFEST")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# ------------------------------------------------------------
# ADD IDs
# ------------------------------------------------------------

for df in [train_df, test_df]:

    df["speaker_id"] = df["utterance_id"].apply(
        extract_speaker_id
    )

    df["video_id"] = df["utterance_id"].apply(
        extract_video_id
    )


# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

all_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ------------------------------------------------------------
# UNIQUE AUDIO IDS
# ------------------------------------------------------------

unique_utterances = (
    all_df["utterance_id"]
    .nunique()
)

unique_speakers = (
    all_df["speaker_id"]
    .nunique()
)

unique_videos = (
    all_df[
        ["speaker_id", "video_id"]
    ]
    .drop_duplicates()
    .shape[0]
)


print("\nSelected utterances:", unique_utterances)
print("Unique speakers:", unique_speakers)
print("Unique videos:", unique_videos)


# ------------------------------------------------------------
# REQUIRED SPEAKER/VIDEO MANIFEST
# ------------------------------------------------------------

video_manifest = (
    all_df[
        [
            "speaker_id",
            "video_id"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["speaker_id", "video_id"]
    )
)


video_manifest_file = (
    OUTPUT_DIR
    / "required_speaker_video_manifest.csv"
)

video_manifest.to_csv(
    video_manifest_file,
    index=False
)


# ------------------------------------------------------------
# UTTERANCE MANIFEST
# ------------------------------------------------------------

utterance_manifest = all_df[
    [
        "utterance_id",
        "speaker_id",
        "video_id",
        "age",
        "age_class",
        "split"
    ]
].copy()


utterance_manifest_file = (
    OUTPUT_DIR
    / "required_utterance_manifest.csv"
)

utterance_manifest.to_csv(
    utterance_manifest_file,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\nSaved:")

print(video_manifest_file)
print(utterance_manifest_file)


print("\n" + "=" * 70)
print("MANIFEST COMPLETE")
print("=" * 70)