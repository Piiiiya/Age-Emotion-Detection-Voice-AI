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

METADATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ravdess_gender"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gender_features"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5.0
MAX_SAMPLES = int(
    SAMPLE_RATE * DURATION
)

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) == 0:
            raise ValueError(
                "Empty audio"
            )

        # ----------------------------------------------------
        # Remove silence
        # ----------------------------------------------------

        y, _ = librosa.effects.trim(
            y,
            top_db=30
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        max_value = np.max(
            np.abs(y)
        )

        if max_value > 0:
            y = y / max_value

        # ----------------------------------------------------
        # Pad / truncate to 5 seconds
        # ----------------------------------------------------

        if len(y) < MAX_SAMPLES:

            y = np.pad(
                y,
                (
                    0,
                    MAX_SAMPLES - len(y)
                )
            )

        else:

            y = y[:MAX_SAMPLES]

        # ====================================================
        # MFCC
        # ====================================================

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mfcc_delta = librosa.feature.delta(
            mfcc
        )

        mfcc_delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # ====================================================
        # MEL SPECTROGRAM
        # ====================================================

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel_delta = librosa.feature.delta(
            mel_db
        )

        mel_delta2 = librosa.feature.delta(
            mel_db,
            order=2
        )

        # ====================================================
        # CHROMA
        # ====================================================

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # ====================================================
        # SPECTRAL FEATURES
        # ====================================================

        spectral_centroid = (
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        spectral_bandwidth = (
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        spectral_rolloff = (
            librosa.feature.spectral_rolloff(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        zero_crossing_rate = (
            librosa.feature.zero_crossing_rate(
                y,
                hop_length=HOP_LENGTH
            )
        )

        rms = (
            librosa.feature.rms(
                y=y,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        spectral_flatness = (
            librosa.feature.spectral_flatness(
                y=y,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        # ====================================================
        # HELPER
        # ====================================================

        def summarize(feature):

            return np.concatenate([
                np.mean(
                    feature,
                    axis=1
                ),

                np.std(
                    feature,
                    axis=1
                ),

                np.min(
                    feature,
                    axis=1
                ),

                np.max(
                    feature,
                    axis=1
                )
            ])

        # ====================================================
        # CREATE FINAL VECTOR
        # ====================================================

        feature_vector = np.concatenate([

            summarize(mfcc),

            summarize(mfcc_delta),

            summarize(mfcc_delta2),

            summarize(mel_db),

            summarize(mel_delta),

            summarize(mel_delta2),

            summarize(chroma),

            summarize(spectral_centroid),

            summarize(spectral_bandwidth),

            summarize(spectral_rolloff),

            summarize(zero_crossing_rate),

            summarize(rms),

            summarize(spectral_flatness)
        ])

        feature_vector = feature_vector.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if np.isnan(feature_vector).any():
            raise ValueError(
                "NaN detected"
            )

        if np.isinf(feature_vector).any():
            raise ValueError(
                "Inf detected"
            )

        return feature_vector

    except Exception as e:

        raise RuntimeError(
            f"{audio_path}: {e}"
        )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING GENDER {split.upper()} SET"
    )
    print("=" * 70)

    metadata_path = (
        METADATA_ROOT
        / f"ravdess_gender_{split}.csv"
    )

    df = pd.read_csv(
        metadata_path
    )

    features = []
    labels = []
    paths = []
    failed = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split}"
    ):

        audio_path = row["path"]

        try:

            feature_vector = extract_features(
                audio_path
            )

            features.append(
                feature_vector
            )

            labels.append(
                0 if row["gender"] == "female"
                else 1
            )

            paths.append(
                audio_path
            )

        except Exception as e:

            failed.append({
                "path": audio_path,
                "error": str(e)
            })

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        OUTPUT_ROOT
        / f"{split}_features.npy",
        features
    )

    np.save(
        OUTPUT_ROOT
        / f"{split}_labels.npy",
        labels
    )

    pd.DataFrame({
        "path": paths
    }).to_csv(
        OUTPUT_ROOT
        / f"{split}_paths.csv",
        index=False
    )

    pd.DataFrame(
        failed
    ).to_csv(
        OUTPUT_ROOT
        / f"{split}_failed.csv",
        index=False
    )

    print(
        f"\n{split.upper()} completed"
    )

    print(
        f"Features shape: {features.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    print(
        f"Failed files: {len(failed)}"
    )

    if len(features) > 0:

        print(
            f"Feature dimension: "
            f"{features.shape[1]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENDER AUDIO FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"\nSample rate: {SAMPLE_RATE}"
    )

    print(
        f"Audio duration: {DURATION} seconds"
    )

    print(
        f"MFCC: {N_MFCC}"
    )

    print(
        f"Mel bands: {N_MELS}"
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        process_split(
            split
        )

    # ========================================================
    # SAVE FEATURE INFORMATION
    # ========================================================

    info = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "label_mapping": {
            "female": 0,
            "male": 1
        }
    }

    with open(
        OUTPUT_ROOT
        / "feature_config.json",
        "w"
    ) as f:

        json.dump(
            info,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("GENDER FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()