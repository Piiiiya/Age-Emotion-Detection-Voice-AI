from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "cremad_splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion_cremad"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

DURATION = 4.0

MAX_SAMPLES = int(
    SAMPLE_RATE * DURATION
)

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256

FMIN = 20

FMAX = 8000


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # Remove NaN / infinite values

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Normalize

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = audio / peak

    # Fixed length

    if len(audio) < MAX_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                MAX_SAMPLES - len(audio),
            ),
            mode="constant",
        )

    else:

        audio = audio[
            :MAX_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio = load_audio(
        file_path
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # Convert to dB

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        log_mel,
        width=9,
        order=1,
    )

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        log_mel,
        width=9,
        order=2,
    )

    # --------------------------------------------------------
    # Standardize each channel independently
    # --------------------------------------------------------

    def standardize(feature):

        mean = np.mean(
            feature
        )

        std = np.std(
            feature
        )

        if std < 1e-8:

            std = 1.0

        return (
            feature - mean
        ) / std


    log_mel = standardize(
        log_mel
    )

    delta = standardize(
        delta
    )

    delta_delta = standardize(
        delta_delta
    )

    # --------------------------------------------------------
    # Stack into 3 channels
    # --------------------------------------------------------

    features = np.stack(
        [
            log_mel,
            delta,
            delta_delta,
        ],
        axis=-1,
    )

    return features.astype(
        np.float32
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    metadata_file,
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {split_name.upper()}"
    )

    print("=" * 70)

    df = pd.read_csv(
        metadata_file
    )

    X = []

    y = []

    successful = 0

    failed = 0

    failures = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name,
    ):

        try:

            relative_path = str(
                row["file"]
            ).replace(
                "\\",
                "/",
            )

            file_path = (
                PROJECT_ROOT
                / relative_path
            )

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Audio not found: {file_path}"
                )

            features = extract_features(
                file_path
            )

            emotion = str(
                row["emotion"]
            ).strip().lower()

            if emotion not in CLASS_TO_INDEX:

                raise ValueError(
                    f"Unknown emotion: {emotion}"
                )

            label = CLASS_TO_INDEX[
                emotion
            ]

            X.append(
                features
            )

            y.append(
                label
            )

            successful += 1

        except Exception as e:

            failed += 1

            failures.append(
                {
                    "file": row.get(
                        "file",
                        "",
                    ),
                    "error": str(e),
                }
            )


    if not X:

        raise RuntimeError(
            f"No samples were successfully processed for {split_name}."
        )


    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / f"X_{split_name}.npy",
        X,
    )

    np.save(
        OUTPUT_DIR
        / f"y_{split_name}.npy",
        y,
    )


    # Save failures if any

    if failures:

        pd.DataFrame(
            failures
        ).to_csv(
            OUTPUT_DIR
            / f"{split_name}_failures.csv",
            index=False,
        )


    print(
        f"\n{split_name.upper()} successful: "
        f"{successful}"
    )

    print(
        f"{split_name.upper()} failed: "
        f"{failed}"
    )

    print(
        f"{split_name.upper()} X shape: "
        f"{X.shape}"
    )

    print(
        f"{split_name.upper()} y shape: "
        f"{y.shape}"
    )


    # Label distribution

    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    print(
        f"\n{split_name.upper()} label distribution:"
    )

    for label, count in zip(
        unique,
        counts,
    ):

        print(
            f"{CLASS_NAMES[label]:10s}: "
            f"{count}"
        )


    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CREMA-D EMOTION FEATURE PREPARATION"
    )

    print("=" * 70)

    print(
        f"\nSample rate: {SAMPLE_RATE}"
    )

    print(
        f"Duration: {DURATION} seconds"
    )

    print(
        f"Mel bins: {N_MELS}"
    )

    print(
        f"N_FFT: {N_FFT}"
    )

    print(
        f"Hop length: {HOP_LENGTH}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    train_X, train_y = process_split(
        "train",
        METADATA_DIR
        / "cremad_male_train.csv",
    )

    val_X, val_y = process_split(
        "val",
        METADATA_DIR
        / "cremad_male_val.csv",
    )

    test_X, test_y = process_split(
        "test",
        METADATA_DIR
        / "cremad_male_test.csv",
    )


    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "classes.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            CLASS_TO_INDEX,
            f,
            indent=4,
        )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL DATASET SUMMARY")

    print("=" * 70)

    print(
        f"\nTRAIN: "
        f"{train_X.shape}"
    )

    print(
        f"VALIDATION: "
        f"{val_X.shape}"
    )

    print(
        f"TEST: "
        f"{test_X.shape}"
    )

    print(
        f"\nClasses:"
    )

    for name, index in CLASS_TO_INDEX.items():

        print(
            f"{index}: {name}"
        )


    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()