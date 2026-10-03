from pathlib import Path
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "age"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "age_features"
METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "common_voice_age"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5.0
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# AGE CLASSES
# ============================================================

AGE_CLASSES = [
    "00_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_plus",
]

CLASS_TO_INDEX = {
    "00_19": 0,
    "20_29": 1,
    "30_39": 2,
    "40_49": 3,
    "50_59": 4,
    "60_plus": 5,
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove silence from beginning/end
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Fixed 5-second length
    if len(audio) < MAX_SAMPLES:

        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:MAX_SAMPLES]

    return audio


def extract_features(file_path):

    audio = load_audio(file_path)

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

    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # MEL
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # SPECTRAL FEATURES
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=HOP_LENGTH
    )

    rms = librosa.feature.rms(
        y=audio,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # STATISTICAL POOLING
    # --------------------------------------------------------

    features = []

    feature_groups = [
        mfcc,
        mfcc_delta,
        mfcc_delta2,
        mel_db,
        chroma,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms,
    ]

    for feature in feature_groups:

        features.extend(
            np.mean(feature, axis=1)
        )

        features.extend(
            np.std(feature, axis=1)
        )

        features.extend(
            np.min(feature, axis=1)
        )

        features.extend(
            np.max(feature, axis=1)
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()} SET")
    print("=" * 70)

    output_features = OUTPUT_ROOT / f"{split_name}_features.npy"
    output_labels = OUTPUT_ROOT / f"{split_name}_labels.npy"
    output_paths = OUTPUT_ROOT / f"{split_name}_paths.csv"

    all_features = []
    all_labels = []
    all_paths = []

    failed_files = []

    # --------------------------------------------------------
    # Collect files
    # --------------------------------------------------------

    files_and_labels = []

    for class_name in AGE_CLASSES:

        class_folder = DATA_ROOT / split_name / class_name

        files = sorted(
            class_folder.glob("*.mp3")
        )

        label = CLASS_TO_INDEX[class_name]

        for file_path in files:

            files_and_labels.append(
                (file_path, label)
            )

    print(
        f"Total files found: {len(files_and_labels):,}"
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    for file_path, label in tqdm(
        files_and_labels,
        desc=f"{split_name}",
        unit="file"
    ):

        try:

            feature_vector = extract_features(
                file_path
            )

            all_features.append(
                feature_vector
            )

            all_labels.append(
                label
            )

            all_paths.append(
                str(file_path)
            )

        except Exception as e:

            failed_files.append(
                {
                    "path": str(file_path),
                    "error": str(e)
                }
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.vstack(
        all_features
    ).astype(np.float32)

    y = np.asarray(
        all_labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_features,
        X
    )

    np.save(
        output_labels,
        y
    )

    pd.DataFrame(
        {
            "path": all_paths,
            "label": y
        }
    ).to_csv(
        output_paths,
        index=False
    )

    # Failed files
    failed_path = (
        OUTPUT_ROOT /
        f"{split_name}_failed.csv"
    )

    pd.DataFrame(
        failed_files
    ).to_csv(
        failed_path,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Successfully processed: {len(X):,}"
    )

    print(
        f"Failed: {len(failed_files):,}"
    )

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Labels: {y.shape}"
    )

    print(
        f"NaN values: {np.isnan(X).sum():,}"
    )

    print(
        f"Infinite values: {np.isinf(X).sum():,}"
    )

    print(
        f"Saved: {output_features}"
    )

    print(
        f"Saved: {output_labels}"
    )

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL AGE FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Sample rate: {SAMPLE_RATE} Hz"
    )

    print(
        f"Fixed duration: {DURATION} seconds"
    )

    print(
        f"Feature dimensions: 804"
    )

    print(
        f"Output directory:\n{OUTPUT_ROOT}"
    )

    print("\nAge classes:")

    for name, index in CLASS_TO_INDEX.items():

        print(
            f"  {index}: {name}"
        )

    print("\nStarting extraction...")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_X, train_y = process_split(
        "train"
    )

    # --------------------------------------------------------
    # DEV
    # --------------------------------------------------------

    dev_X, dev_y = process_split(
        "dev"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_X, test_y = process_split(
        "test"
    )

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    classes_path = (
        OUTPUT_ROOT /
        "age_classes.json"
    )

    with open(
        classes_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            CLASS_TO_INDEX,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Train: {train_X.shape}"
    )

    print(
        f"Dev:   {dev_X.shape}"
    )

    print(
        f"Test:  {test_X.shape}"
    )

    print(
        f"\nClass mapping saved to:"
    )

    print(classes_path)

    print("=" * 70)


if __name__ == "__main__":
    main()