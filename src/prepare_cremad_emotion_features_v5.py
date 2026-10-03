import os
import json
import warnings

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METADATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "cremad_splits"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm_v5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SR = 16000
DURATION = 4
TARGET_LENGTH = SR * DURATION

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad"
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# FEATURE STATISTICS
# ============================================================

def statistics(feature):
    """
    Calculate robust statistics along time axis.
    """

    return np.concatenate([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
        np.min(feature, axis=1),
        np.max(feature, axis=1),
        np.median(feature, axis=1)
    ])


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    # Remove NaN / Inf
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize amplitude
    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:
        audio = audio / max_value

    # Fixed 4 seconds
    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                TARGET_LENGTH - len(audio)
            )
        )

    else:

        audio = audio[:TARGET_LENGTH]

    return audio


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(path):

    audio = load_audio(path)

    features = []


    # --------------------------------------------------------
    # 1. MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    features.append(
        statistics(mfcc)
    )


    # --------------------------------------------------------
    # 2. MFCC DELTA
    # --------------------------------------------------------

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    features.append(
        statistics(mfcc_delta)
    )


    # --------------------------------------------------------
    # 3. MFCC DELTA-DELTA
    # --------------------------------------------------------

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.append(
        statistics(mfcc_delta2)
    )


    # --------------------------------------------------------
    # 4. MEL SPECTROGRAM
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
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

    features.append(
        statistics(mel_db)
    )


    # --------------------------------------------------------
    # 5. MEL DELTA
    # --------------------------------------------------------

    mel_delta = librosa.feature.delta(
        mel_db
    )

    features.append(
        statistics(mel_delta)
    )


    # --------------------------------------------------------
    # 6. MEL DELTA-DELTA
    # --------------------------------------------------------

    mel_delta2 = librosa.feature.delta(
        mel_db,
        order=2
    )

    features.append(
        statistics(mel_delta2)
    )


    # --------------------------------------------------------
    # 7. CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(chroma)
    )


    # --------------------------------------------------------
    # 8. SPECTRAL CONTRAST
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(contrast)
    )


    # --------------------------------------------------------
    # 9. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(zcr)
    )


    # --------------------------------------------------------
    # 10. RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(rms)
    )


    # --------------------------------------------------------
    # 11. SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(centroid)
    )


    # --------------------------------------------------------
    # 12. SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(bandwidth)
    )


    # --------------------------------------------------------
    # 13. SPECTRAL ROLLOFF
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(rolloff)
    )


    # --------------------------------------------------------
    # 14. SPECTRAL FLATNESS
    # --------------------------------------------------------

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(
        statistics(flatness)
    )


    # --------------------------------------------------------
    # 15. RMS DELTA
    # --------------------------------------------------------

    rms_delta = librosa.feature.delta(
        rms
    )

    features.append(
        statistics(rms_delta)
    )


    # --------------------------------------------------------
    # 16. RMS DELTA-DELTA
    # --------------------------------------------------------

    rms_delta2 = librosa.feature.delta(
        rms,
        order=2
    )

    features.append(
        statistics(rms_delta2)
    )


    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    feature_vector = np.concatenate(
        features
    )

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return feature_vector.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    csv_name,
    output_name
):

    csv_path = os.path.join(
        METADATA_DIR,
        csv_name
    )

    print("\n")
    print("=" * 70)
    print(
        f"Processing {csv_name}"
    )
    print("=" * 70)

    df = pd.read_csv(
        csv_path
    )

    X = []
    y = []

    successful = 0
    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        file_path = row["file"]

        emotion = row["emotion"]

        try:

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    file_path
                )

            feature_vector = extract_features(
                file_path
            )

            X.append(
                feature_vector
            )

            y.append(
                CLASS_TO_ID[emotion]
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFailed: {file_path}"
            )

            print(
                f"Error: {e}"
            )


    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )


    print("\n")
    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )


    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{output_name}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{output_name}.npy"
        ),
        y
    )

    return X, y


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREMA-D EMOTION FEATURE EXTRACTION V5")
    print("=" * 70)

    print(
        "\nFeature configuration:"
    )

    print(
        f"Sample rate: {SR}"
    )

    print(
        f"Duration: {DURATION} seconds"
    )

    print(
        f"MFCC: {N_MFCC}"
    )

    print(
        f"Mel bands: {N_MELS}"
    )

    print(
        "\nStarting extraction..."
    )


    X_train, y_train = process_split(
        "cremad_male_train.csv",
        "train"
    )


    X_val, y_val = process_split(
        "cremad_male_val.csv",
        "val"
    )


    X_test, y_test = process_split(
        "cremad_male_test.csv",
        "test"
    )


    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    classes_path = os.path.join(
        OUTPUT_DIR,
        "classes.json"
    )

    with open(
        classes_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            CLASS_TO_ID,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("V5 FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        "\nTrain:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles created:"
    )

    print(
        "X_train.npy"
    )

    print(
        "y_train.npy"
    )

    print(
        "X_val.npy"
    )

    print(
        "y_val.npy"
    )

    print(
        "X_test.npy"
    )

    print(
        "y_test.npy"
    )

    print(
        "classes.json"
    )

    print(
        "\n"
        + "=" * 70
    )