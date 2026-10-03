import os
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

METADATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "emotion_v3"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# METADATA FILES
# ============================================================

METADATA_FILES = {
    "train": "ravdess_male_train.csv",
    "validation": "ravdess_male_val.csv",
    "test": "ravdess_male_test.csv"
}


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

DURATION = 4.0

TARGET_LENGTH = int(
    SAMPLE_RATE * DURATION
)


# ============================================================
# FEATURE SETTINGS
# ============================================================

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256

EPSILON = 1e-10


# ============================================================
# EMOTION CLASSES
# ============================================================

CLASS_NAMES = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}


# ============================================================
# RESOLVE AUDIO PATH
# ============================================================

def resolve_audio_path(relative_path):

    if not isinstance(
        relative_path,
        str
    ):
        return None

    relative_path = relative_path.strip()

    # Convert Windows separators
    relative_path = relative_path.replace(
        "/",
        os.sep
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # CSV contains:
    #
    # ..\data\raw\ravdess\extracted\Actor_01\file.wav
    #
    # Because the CSV is relative to the project root,
    # remove the "..\" before joining with PROJECT_ROOT.
    # --------------------------------------------------------

    while relative_path.startswith(
        ".." + os.sep
    ):

        relative_path = relative_path[
            3:
        ]

    candidate = os.path.normpath(
        os.path.join(
            PROJECT_ROOT,
            relative_path
        )
    )

    if os.path.exists(candidate):

        return candidate

    # --------------------------------------------------------
    # Direct RAVDESS search fallback
    # --------------------------------------------------------

    filename = os.path.basename(
        relative_path
    )

    ravdess_root = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "ravdess",
        "extracted"
    )

    if os.path.exists(
        ravdess_root
    ):

        for root, dirs, files in os.walk(
            ravdess_root
        ):

            if filename in files:

                return os.path.join(
                    root,
                    filename
                )

    return None


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    y = y.astype(
        np.float32
    )

    # Remove DC offset
    y = y - np.mean(y)

    # Normalize amplitude
    max_amplitude = np.max(
        np.abs(y)
    )

    if max_amplitude > 0:

        y = y / max_amplitude

    # --------------------------------------------------------
    # Make exactly 4 seconds
    # --------------------------------------------------------

    if len(y) < TARGET_LENGTH:

        y = np.pad(
            y,
            (
                0,
                TARGET_LENGTH - len(y)
            ),
            mode="constant"
        )

    else:

        y = y[
            :TARGET_LENGTH
        ]

    return y


# ============================================================
# STANDARDIZE FEATURE
# ============================================================

def standardize_feature(
    feature
):

    mean = np.mean(
        feature
    )

    std = np.std(
        feature
    )

    feature = (
        feature - mean
    ) / (
        std + EPSILON
    )

    return feature.astype(
        np.float32
    )


# ============================================================
# EXTRACT V3 FEATURES
# ============================================================

def extract_features(
    audio_path
):

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # --------------------------------------------------------
    # Log-Mel
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    log_mel = standardize_feature(
        log_mel
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        log_mel,
        width=9,
        order=1
    )

    delta = standardize_feature(
        delta
    )

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        log_mel,
        width=9,
        order=2
    )

    delta_delta = standardize_feature(
        delta_delta
    )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    if not (
        log_mel.shape
        == delta.shape
        == delta_delta.shape
    ):

        raise ValueError(
            "Feature shapes do not match.\n"
            f"Log-Mel: {log_mel.shape}\n"
            f"Delta: {delta.shape}\n"
            f"Delta-Delta: {delta_delta.shape}"
        )

    # --------------------------------------------------------
    # Stack channels
    # --------------------------------------------------------

    features = np.stack(
        [
            log_mel,
            delta,
            delta_delta
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # Validate final shape
    # --------------------------------------------------------

    if features.ndim != 3:

        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    if features.shape[0] != N_MELS:

        raise ValueError(
            f"Unexpected height: "
            f"{features.shape}"
        )

    if features.shape[2] != 3:

        raise ValueError(
            f"Expected 3 channels: "
            f"{features.shape}"
        )

    return features.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name
):

    metadata_filename = (
        METADATA_FILES[
            split_name
        ]
    )

    metadata_path = os.path.join(
        METADATA_DIR,
        metadata_filename
    )

    if not os.path.exists(
        metadata_path
    ):

        raise FileNotFoundError(
            f"\nMetadata file not found:\n"
            f"{metadata_path}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        metadata_path
    )

    print()
    print("=" * 60)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print(
        f"Metadata: {metadata_filename}"
    )
    print(
        f"Files: {len(df)}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "file",
        "emotion_code",
        "emotion"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' "
                f"not found in {metadata_filename}"
            )

    X = []

    y = []

    successful = 0

    failed = 0

    failed_files = []

    # --------------------------------------------------------
    # Process every row
    # --------------------------------------------------------

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name.upper()
    ):

        try:

            # ------------------------------------------------
            # Resolve audio
            # ------------------------------------------------

            original_path = row[
                "file"
            ]

            audio_path = resolve_audio_path(
                original_path
            )

            if audio_path is None:

                raise FileNotFoundError(
                    f"Audio file not found: "
                    f"{original_path}"
                )

            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            features = extract_features(
                audio_path
            )

            # ------------------------------------------------
            # Get emotion code
            #
            # CSV:
            # 1 = neutral
            # 2 = calm
            # 3 = happy
            # 4 = sad
            # 5 = angry
            # 6 = fearful
            # 7 = disgust
            # 8 = surprised
            #
            # Our model labels:
            # 0 = angry
            # 1 = calm
            # 2 = disgust
            # 3 = fearful
            # 4 = happy
            # 5 = neutral
            # 6 = sad
            # 7 = surprised
            # ------------------------------------------------

            ravdess_code = int(
                row[
                    "emotion_code"
                ]
            )

            emotion_name = str(
                row[
                    "emotion"
                ]
            ).strip().lower()

            # ------------------------------------------------
            # Map using emotion name
            # This avoids incorrect numerical mapping.
            # ------------------------------------------------

            emotion_to_label = {
                "angry": 0,
                "calm": 1,
                "disgust": 2,
                "fearful": 3,
                "happy": 4,
                "neutral": 5,
                "sad": 6,
                "surprised": 7
            }

            if emotion_name not in emotion_to_label:

                raise ValueError(
                    f"Unknown emotion: "
                    f"{emotion_name}"
                )

            label = emotion_to_label[
                emotion_name
            ]

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            X.append(
                features
            )

            y.append(
                label
            )

            successful += 1

        except Exception as e:

            failed += 1

            failed_files.append(
                {
                    "index": int(index),
                    "file": str(
                        row["file"]
                    ),
                    "error": str(e)
                }
            )

    # --------------------------------------------------------
    # Convert arrays
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
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"{split_name.upper()} successful: "
        f"{successful}"
    )

    print(
        f"{split_name.upper()} failed: "
        f"{failed}"
    )

    print(
        f"{split_name.upper()} X: "
        f"{X.shape}"
    )

    print(
        f"{split_name.upper()} y: "
        f"{y.shape}"
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{split_name}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{split_name}.npy"
        ),
        y
    )

    # --------------------------------------------------------
    # Save failed files
    # --------------------------------------------------------

    if failed_files:

        failed_path = os.path.join(
            OUTPUT_DIR,
            f"{split_name}_failed.json"
        )

        with open(
            failed_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                failed_files,
                f,
                indent=4
            )

    return X, y


# ============================================================
# PRINT LABEL DISTRIBUTION
# ============================================================

def print_label_distribution(
    y,
    split_name
):

    print()

    print(
        f"{split_name.upper()} "
        f"emotion distribution:"
    )

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        emotion_name = CLASS_NAMES.get(
            int(label),
            "unknown"
        )

        print(
            f"  {int(label)} -> "
            f"{emotion_name:<10} "
            f"{int(count)}"
        )


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

def save_class_information():

    class_path = os.path.join(
        OUTPUT_DIR,
        "classes.json"
    )

    with open(
        class_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            CLASS_NAMES,
            f,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "V3 EMOTION FEATURE PREPARATION"
    )
    print("=" * 60)

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Metadata directory:"
    )

    print(
        METADATA_DIR
    )

    print()

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    print()

    print(
        "Checking metadata files..."
    )

    for split_name, filename in (
        METADATA_FILES.items()
    ):

        path = os.path.join(
            METADATA_DIR,
            filename
        )

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"\nMissing metadata:\n{path}"
            )

        df = pd.read_csv(
            path
        )

        print(
            f"  {split_name.capitalize():<12} "
            f"{len(df)} files -> "
            f"{filename}"
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    X_train, y_train = process_split(
        "train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    X_val, y_val = process_split(
        "validation"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    X_test, y_test = process_split(
        "test"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print_label_distribution(
        y_train,
        "train"
    )

    print_label_distribution(
        y_val,
        "validation"
    )

    print_label_distribution(
        y_test,
        "test"
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    save_class_information()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print("=" * 60)
    print(
        "V3 FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 60)

    print()

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"y_val:   {y_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    print()

    print(
        "Feature channels:"
    )

    print(
        "  Channel 1 -> Log-Mel Spectrogram"
    )

    print(
        "  Channel 2 -> Delta Log-Mel"
    )

    print(
        "  Channel 3 -> Delta-Delta Log-Mel"
    )

    print()

    print(
        "Expected individual sample shape:"
    )

    print(
        "(128, 251, 3)"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()