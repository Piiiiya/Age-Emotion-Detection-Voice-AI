from pathlib import Path
import json

import numpy as np
import pandas as pd
import librosa


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4

N_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_MFCC = 40

N_FFT = 1024
HOP_LENGTH = 256

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
    for index, emotion
    in enumerate(EMOTION_CLASSES)
}


# ============================================================
# METADATA FILES
# ============================================================

TRAIN_CSV = (
    METADATA_DIR
    / "ravdess_male_train.csv"
)

VAL_CSV = (
    METADATA_DIR
    / "ravdess_male_val.csv"
)

TEST_CSV = (
    METADATA_DIR
    / "ravdess_male_test.csv"
)


# ============================================================
# SAVE CLASS MAPPING
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
# RESOLVE AUDIO PATH
# ============================================================

def resolve_audio_path(file_path):

    path_string = str(
        file_path
    ).replace("\\", "/")

    # Absolute path
    path = Path(path_string)

    if path.is_absolute() and path.exists():
        return path.resolve()

    # Relative to project root
    candidate_1 = (
        PROJECT_ROOT
        / path_string
    ).resolve()

    if candidate_1.exists():
        return candidate_1

    # Relative to notebooks directory
    candidate_2 = (
        PROJECT_ROOT
        / "notebooks"
        / path_string
    ).resolve()

    if candidate_2.exists():
        return candidate_2

    # Directly locate data/raw portion
    marker = "data/raw/"

    lower_path = path_string.lower()

    if marker in lower_path:

        index = lower_path.find(
            marker
        )

        relative_path = path_string[index:]

        candidate_3 = (
            PROJECT_ROOT
            / relative_path
        ).resolve()

        if candidate_3.exists():
            return candidate_3

    return candidate_2


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Fixed length
    if len(audio) < N_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                N_SAMPLES - len(audio)
            )
        )

    else:

        audio = audio[:N_SAMPLES]

    # Normalize amplitude
    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio
            / max_value
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# CREATE 3-CHANNEL FEATURES
# ============================================================

def extract_features(file_path):

    audio = load_audio(
        file_path
    )

    # --------------------------------------------------------
    # Log-Mel
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Standardize
    mel_db = (
        mel_db
        - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc = (
        mfcc
        - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    # Resize MFCC to same height as Mel
    mfcc = librosa.util.fix_length(
        mfcc,
        size=mel_db.shape[1],
        axis=1
    )

    mfcc_resized = np.resize(
        mfcc,
        (
            N_MELS,
            mel_db.shape[1]
        )
    )

    # --------------------------------------------------------
    # Delta MFCC
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    delta = (
        delta
        - np.mean(delta)
    ) / (
        np.std(delta) + 1e-8
    )

    delta = librosa.util.fix_length(
        delta,
        size=mel_db.shape[1],
        axis=1
    )

    delta_resized = np.resize(
        delta,
        (
            N_MELS,
            mel_db.shape[1]
        )
    )

    # --------------------------------------------------------
    # Stack into 3 channels
    # --------------------------------------------------------

    features = np.stack(
        [
            mel_db,
            mfcc_resized,
            delta_resized
        ],
        axis=-1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    dataframe,
    split_name
):

    X = []
    y = []

    total = len(
        dataframe
    )

    print()
    print("=" * 60)
    print(
        f"PROCESSING {split_name}"
    )
    print(
        f"Files: {total}"
    )
    print("=" * 60)

    successful = 0
    failed = 0

    for index, row in dataframe.reset_index(
        drop=True
    ).iterrows():

        file_path = resolve_audio_path(
            row["file"]
        )

        emotion = row["emotion"]

        try:

            if not file_path.exists():

                raise FileNotFoundError(
                    file_path
                )

            features = extract_features(
                file_path
            )

            X.append(
                features
            )

            y.append(
                EMOTION_TO_INDEX[
                    emotion
                ]
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR: {file_path}"
            )

            print(e)

        if (
            (index + 1) % 25 == 0
            or index == 0
            or index + 1 == total
        ):

            print(
                f"Processed {index + 1}/{total}"
                f" | Successful: {successful}"
                f" | Failed: {failed}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print()
    print(
        f"{split_name} successful:",
        successful
    )

    print(
        f"{split_name} failed:",
        failed
    )

    print(
        f"{split_name} X:",
        X.shape
    )

    print(
        f"{split_name} y:",
        y.shape
    )

    return X, y


# ============================================================
# LOAD CSV FILES
# ============================================================

print(
    "Loading metadata..."
)

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# PROCESS
# ============================================================

X_train, y_train = process_split(
    train_df,
    "TRAIN"
)

X_val, y_val = process_split(
    val_df,
    "VALIDATION"
)

X_test, y_test = process_split(
    test_df,
    "TEST"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(X_train) == 0:
    raise RuntimeError(
        "Training dataset is empty."
    )

if len(X_val) == 0:
    raise RuntimeError(
        "Validation dataset is empty."
    )

if len(X_test) == 0:
    raise RuntimeError(
        "Test dataset is empty."
    )


# ============================================================
# SAVE
# ============================================================

print()
print(
    "Saving V2 datasets..."
)

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
# FINAL
# ============================================================

print()
print("=" * 60)
print("V2 FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

print()
print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)