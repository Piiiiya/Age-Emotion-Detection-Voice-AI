from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAVDESS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ravdess"
    / "extracted"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ravdess_gender"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SPEAKER-INDEPENDENT SPLIT
# ============================================================

# 24 actors total
#
# Train: 16 actors
# Validation: 4 actors
# Test: 4 actors
#
# Each split contains both male and female actors.

TRAIN_ACTORS = [
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
]

VAL_ACTORS = [
    17, 18, 19, 20
]

TEST_ACTORS = [
    21, 22, 23, 24
]


# ============================================================
# GENDER MAPPING
# ============================================================

# RAVDESS actor IDs:
#
# Odd  = male
# Even = female

def get_gender(actor_id):

    if actor_id % 2 == 1:
        return "male"

    return "female"


# ============================================================
# FIND AUDIO FILES
# ============================================================

print("=" * 70)
print("RAVDESS GENDER DATASET PREPARATION")
print("=" * 70)

print("\nRAVDESS root:")
print(RAVDESS_ROOT)


audio_files = sorted(
    RAVDESS_ROOT.rglob("*.wav")
)

print(
    f"\nTotal WAV files found: "
    f"{len(audio_files)}"
)


# ============================================================
# BUILD RECORDS
# ============================================================

records = []

for path in audio_files:

    # Actor folder is something like:
    # Actor_01
    # Actor_02
    # etc.

    actor_folder = path.parent.name

    if not actor_folder.startswith("Actor_"):
        continue

    actor_id = int(
        actor_folder.replace(
            "Actor_",
            ""
        )
    )

    gender = get_gender(actor_id)

    # RAVDESS filename:
    #
    # 03-01-01-01-01-01-01.wav
    #
    # The first field identifies modality.
    # We only need the file path, actor and gender here.

    filename_parts = path.stem.split("-")

    records.append({
        "path": str(path),
        "filename": path.name,
        "actor_id": actor_id,
        "gender": gender,
        "emotion_id": int(filename_parts[2]),
    })


df = pd.DataFrame(records)


# ============================================================
# ASSIGN SPLITS
# ============================================================

def assign_split(actor_id):

    if actor_id in TRAIN_ACTORS:
        return "train"

    if actor_id in VAL_ACTORS:
        return "val"

    if actor_id in TEST_ACTORS:
        return "test"

    return None


df["split"] = df["actor_id"].apply(
    assign_split
)

df = df.dropna(
    subset=["split"]
)


# ============================================================
# DISPLAY DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"\nTotal usable files: "
    f"{len(df)}"
)

print(
    f"Unique actors: "
    f"{df['actor_id'].nunique()}"
)

print("\nGender distribution:")

print(
    df["gender"]
    .value_counts()
    .sort_index()
)


print("\nSplit distribution:")

print(
    df["split"]
    .value_counts()
)


print("\nGender by split:")

print(
    pd.crosstab(
        df["split"],
        df["gender"]
    )
)


print("\nActors by split:")

for split in ["train", "val", "test"]:

    actors = sorted(
        df.loc[
            df["split"] == split,
            "actor_id"
        ].unique()
    )

    print(
        f"{split:5s}: "
        f"{actors}"
    )


# ============================================================
# VERIFY NO SPEAKER LEAKAGE
# ============================================================

train_actors = set(
    df.loc[
        df["split"] == "train",
        "actor_id"
    ]
)

val_actors = set(
    df.loc[
        df["split"] == "val",
        "actor_id"
    ]
)

test_actors = set(
    df.loc[
        df["split"] == "test",
        "actor_id"
    ]
)


assert train_actors.isdisjoint(
    val_actors
)

assert train_actors.isdisjoint(
    test_actors
)

assert val_actors.isdisjoint(
    test_actors
)

print(
    "\nSpeaker leakage check: PASSED"
)


# ============================================================
# SAVE COMPLETE METADATA
# ============================================================

all_path = (
    OUTPUT_ROOT /
    "ravdess_gender_all.csv"
)

df.to_csv(
    all_path,
    index=False
)


# ============================================================
# SAVE SPLITS
# ============================================================

for split in ["train", "val", "test"]:

    split_df = df[
        df["split"] == split
    ].copy()

    output_path = (
        OUTPUT_ROOT /
        f"ravdess_gender_{split}.csv"
    )

    split_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved {split}:"
    )

    print(output_path)

    print(
        f"Samples: "
        f"{len(split_df)}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL GENDER DATASET")
print("=" * 70)

for split in ["train", "val", "test"]:

    split_df = df[
        df["split"] == split
    ]

    male = (
        split_df["gender"]
        .eq("male")
        .sum()
    )

    female = (
        split_df["gender"]
        .eq("female")
        .sum()
    )

    print(
        f"\n{split.upper()}"
    )

    print(
        f"  Total : {len(split_df)}"
    )

    print(
        f"  Male  : {male}"
    )

    print(
        f"  Female: {female}"
    )


print("\n" + "=" * 70)
print("RAVDESS GENDER PREPARATION COMPLETE")
print("=" * 70)