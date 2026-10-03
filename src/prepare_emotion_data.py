from pathlib import Path
import json

import numpy as np
import pandas as pd

from audio_features import audio_to_log_mel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "emotion"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METADATA FILES
# ============================================================

TRAIN_CSV = METADATA_DIR / "ravdess_male_train.csv"
VAL_CSV = METADATA_DIR / "ravdess_male_val.csv"
TEST_CSV = METADATA_DIR / "ravdess_male_test.csv"


# ============================================================
# EMOTION CLASSES
# ============================================================

EMOTION_CLASSES = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

EMOTION_TO_INDEX = {
    emotion: index
    for index, emotion in enumerate(EMOTION_CLASSES)
}

print("Emotion classes:")
print(EMOTION_TO_INDEX)


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

with open(
    OUTPUT_DIR / "emotion_classes.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        EMOTION_TO_INDEX,
        f,
        indent=4
    )


# ============================================================
# FIX METADATA AUDIO PATH
# ============================================================

def resolve_audio_path(file_path):
    """
    Resolve the audio path stored inside the metadata CSV.

    The metadata was originally created from the notebooks
    directory, therefore paths such as:

        ../data/raw/...

    are relative to the notebooks directory.

    This function converts them into valid absolute paths.
    """

    path = Path(str(file_path))

    # --------------------------------------------------------
    # If the path is already absolute
    # --------------------------------------------------------

    if path.is_absolute():

        if path.exists():
            return path.resolve()

    # --------------------------------------------------------
    # Convert Windows backslashes consistently
    # --------------------------------------------------------

    path_string = str(file_path).replace("\\", "/")

    # --------------------------------------------------------
    # First possibility:
    # Path relative to project root
    # --------------------------------------------------------

    candidate_1 = (
        PROJECT_ROOT / path_string
    ).resolve()

    if candidate_1.exists():
        return candidate_1

    # --------------------------------------------------------
    # Second possibility:
    # Path relative to notebooks directory
    #
    # This handles:
    #
    # ../data/raw/ravdess/...
    # --------------------------------------------------------

    candidate_2 = (
        PROJECT_ROOT / "notebooks" / path_string
    ).resolve()

    if candidate_2.exists():
        return candidate_2

    # --------------------------------------------------------
    # Third possibility:
    # Extract the data/raw portion directly.
    #
    # This makes the script robust against different
    # relative-path formats.
    # --------------------------------------------------------

    marker = "data/raw/"

    if marker in path_string.lower():

        index = path_string.lower().find(marker)

        relative_data_path = path_string[index:]

        candidate_3 = (
            PROJECT_ROOT / relative_data_path
        ).resolve()

        if candidate_3.exists():
            return candidate_3

    # --------------------------------------------------------
    # If nothing worked, return the best candidate.
    # The caller will report the missing file.
    # --------------------------------------------------------

    return candidate_2


# ============================================================
# TEST ONE AUDIO PATH BEFORE PROCESSING EVERYTHING
# ============================================================

def verify_sample_path(df, split_name):

    if len(df) == 0:
        return

    original_path = df.iloc[0]["file"]

    resolved_path = resolve_audio_path(
        original_path
    )

    print()
    print("=" * 60)
    print(f"{split_name} PATH VERIFICATION")
    print("=" * 60)

    print("Original path:")
    print(original_path)

    print()
    print("Resolved path:")
    print(resolved_path)

    print()
    print("File exists:", resolved_path.exists())

    if not resolved_path.exists():

        raise FileNotFoundError(
            f"\nAudio file could not be found:\n"
            f"{resolved_path}"
        )


# ============================================================
# PROCESS DATAFRAME
# ============================================================

def process_dataframe(df, split_name):

    X = []
    y = []

    total = len(df)

    print()
    print("=" * 60)
    print(f"Processing {split_name} dataset")
    print(f"Files: {total}")
    print("=" * 60)

    successful = 0
    failed = 0

    for index, row in df.reset_index(drop=True).iterrows():

        original_file_path = row["file"]

        file_path = resolve_audio_path(
            original_file_path
        )

        emotion = row["emotion"]

        try:

            if not file_path.exists():

                raise FileNotFoundError(
                    f"File does not exist: {file_path}"
                )

            mel = audio_to_log_mel(
                file_path
            )

            X.append(mel)

            y.append(
                EMOTION_TO_INDEX[emotion]
            )

            successful += 1

        except Exception as e:

            failed += 1

            print()
            print(
                f"ERROR processing file {index + 1}/{total}:"
            )

            print(file_path)
            print(e)

        # Progress
        if (
            (index + 1) % 25 == 0
            or index == 0
            or index + 1 == total
        ):

            print(
                f"Processed {index + 1}/{total} "
                f"| Successful: {successful} "
                f"| Failed: {failed}"
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Add CNN channel dimension
    # --------------------------------------------------------

    if len(X) > 0:

        X = X[..., np.newaxis]

    print()
    print(f"{split_name} successful:", successful)
    print(f"{split_name} failed:", failed)
    print(f"{split_name} X shape:", X.shape)
    print(f"{split_name} y shape:", y.shape)

    return X, y


# ============================================================
# LOAD METADATA
# ============================================================

print()
print("=" * 60)
print("LOADING METADATA")
print("=" * 60)

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

print("Train records:", len(train_df))
print("Validation records:", len(val_df))
print("Test records:", len(test_df))


# ============================================================
# VERIFY PATHS BEFORE PROCESSING
# ============================================================

verify_sample_path(
    train_df,
    "TRAIN"
)

verify_sample_path(
    val_df,
    "VALIDATION"
)

verify_sample_path(
    test_df,
    "TEST"
)


# ============================================================
# PROCESS DATASETS
# ============================================================

X_train, y_train = process_dataframe(
    train_df,
    "TRAIN"
)

X_val, y_val = process_dataframe(
    val_df,
    "VALIDATION"
)

X_test, y_test = process_dataframe(
    test_df,
    "TEST"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(X_train) == 0:

    raise RuntimeError(
        "Training dataset is empty. "
        "Check the audio paths."
    )

if len(X_val) == 0:

    raise RuntimeError(
        "Validation dataset is empty. "
        "Check the audio paths."
    )

if len(X_test) == 0:

    raise RuntimeError(
        "Test dataset is empty. "
        "Check the audio paths."
    )


# ============================================================
# SAVE NUMPY ARRAYS
# ============================================================

print()
print("=" * 60)
print("SAVING PROCESSED DATASETS")
print("=" * 60)

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print()
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print()
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print()
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print()
print("Saved to:")
print(OUTPUT_DIR)