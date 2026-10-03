from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "crema_d_male_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "cremad_splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREMA-D SPEAKER-INDEPENDENT SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal recordings: {len(df)}")
print(f"Total actors: {df['actor_id'].nunique()}")


# ============================================================
# ACTOR INFORMATION
# ============================================================

actors = (
    df[
        ["actor_id", "age"]
    ]
    .drop_duplicates()
    .sort_values("actor_id")
    .reset_index(drop=True)
)

print("\nActor summary:")
print(actors.to_string(index=False))


# ============================================================
# FIXED SPEAKER SPLIT
# ============================================================
#
# 48 actors total
#
# TRAIN      34 actors
# VALIDATION  7 actors
# TEST        7 actors
#
# IMPORTANT:
# Test contains two unseen senior speakers:
#   1016 -> 61
#   1034 -> 74
#
# Validation contains:
#   1050 -> 62
#
# Training contains:
#   1067 -> 66
#   1087 -> 62
#
# This keeps senior voices represented across the
# development/evaluation pipeline without putting
# all senior speakers into training.
# ============================================================

TEST_ACTORS = [
    1016,
    1034,
    1005,
    1027,
    1048,
    1069,
    1088,
]

VAL_ACTORS = [
    1050,
    1017,
    1036,
    1044,
    1071,
    1083,
    1090,
]


# ============================================================
# TRAIN ACTORS
# ============================================================

all_actors = set(
    actors["actor_id"].tolist()
)

test_set = set(TEST_ACTORS)
val_set = set(VAL_ACTORS)

train_set = (
    all_actors
    - test_set
    - val_set
)

train_actors = sorted(train_set)
val_actors = sorted(val_set)
test_actors = sorted(test_set)


# ============================================================
# SAFETY CHECKS
# ============================================================

assert len(train_actors) == 34
assert len(val_actors) == 7
assert len(test_actors) == 7

assert (
    set(train_actors)
    .isdisjoint(val_actors)
)

assert (
    set(train_actors)
    .isdisjoint(test_actors)
)

assert (
    set(val_actors)
    .isdisjoint(test_actors)
)

assert (
    set(train_actors)
    | set(val_actors)
    | set(test_actors)
) == all_actors


# ============================================================
# CREATE DATASETS
# ============================================================

train_df = df[
    df["actor_id"].isin(train_actors)
].copy()

val_df = df[
    df["actor_id"].isin(val_actors)
].copy()

test_df = df[
    df["actor_id"].isin(test_actors)
].copy()


# ============================================================
# SORT
# ============================================================

train_df = train_df.sort_values(
    ["actor_id", "filename"]
).reset_index(drop=True)

val_df = val_df.sort_values(
    ["actor_id", "filename"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["actor_id", "filename"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

train_file = (
    OUTPUT_DIR
    / "cremad_male_train.csv"
)

val_file = (
    OUTPUT_DIR
    / "cremad_male_val.csv"
)

test_file = (
    OUTPUT_DIR
    / "cremad_male_test.csv"
)


train_df.to_csv(
    train_file,
    index=False
)

val_df.to_csv(
    val_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print("\nTRAIN")
print(f"Actors: {len(train_actors)}")
print(f"Recordings: {len(train_df)}")
print(f"Actors: {train_actors}")

print("\nVALIDATION")
print(f"Actors: {len(val_actors)}")
print(f"Recordings: {len(val_df)}")
print(f"Actors: {val_actors}")

print("\nTEST")
print(f"Actors: {len(test_actors)}")
print(f"Recordings: {len(test_df)}")
print(f"Actors: {test_actors}")


# ============================================================
# AGE INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("TEST ACTOR AGES")
print("-" * 70)

print(
    test_df[
        ["actor_id", "age"]
    ]
    .drop_duplicates()
    .sort_values("actor_id")
    .to_string(index=False)
)


print("\n" + "-" * 70)
print("VALIDATION ACTOR AGES")
print("-" * 70)

print(
    val_df[
        ["actor_id", "age"]
    ]
    .drop_duplicates()
    .sort_values("actor_id")
    .to_string(index=False)
)


# ============================================================
# EMOTION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TRAIN EMOTION DISTRIBUTION")
print("-" * 70)

print(
    train_df["emotion"]
    .value_counts()
    .sort_index()
)


print("\n" + "-" * 70)
print("VALIDATION EMOTION DISTRIBUTION")
print("-" * 70)

print(
    val_df["emotion"]
    .value_counts()
    .sort_index()
)


print("\n" + "-" * 70)
print("TEST EMOTION DISTRIBUTION")
print("-" * 70)

print(
    test_df["emotion"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SPEAKER LEAKAGE CHECK
# ============================================================

print("\n" + "-" * 70)
print("SPEAKER LEAKAGE CHECK")
print("-" * 70)

train_ids = set(train_df["actor_id"])
val_ids = set(val_df["actor_id"])
test_ids = set(test_df["actor_id"])

print(
    "Train ∩ Validation:",
    train_ids & val_ids
)

print(
    "Train ∩ Test:",
    train_ids & test_ids
)

print(
    "Validation ∩ Test:",
    val_ids & test_ids
)


# ============================================================
# OUTPUT
# ============================================================

print("\nOutput directory:")

print(OUTPUT_DIR)

print("\nCreated files:")

print(train_file)
print(val_file)
print(test_file)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)