from pathlib import Path

import joblib
import librosa
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "age"
)

AGE_MODEL_PATH = (
    MODEL_ROOT
    / "age_linear_svm.joblib"
)

SCALER_PATH = (
    MODEL_ROOT
    / "age_svm_scaler.joblib"
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
# AGE CLASSES
# ============================================================

AGE_CLASSES = {
    0: "00_19",
    1: "20_29",
    2: "30_39",
    3: "40_49",
    4: "50_59",
    5: "60_plus"
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

age_model = joblib.load(
    AGE_MODEL_PATH
)

age_scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

def summarize(feature):

    return np.concatenate([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
        np.min(feature, axis=1),
        np.max(feature, axis=1)
    ])


# ============================================================
# EXACT 804-DIMENSIONAL FEATURE EXTRACTION
# ============================================================

def extract_age_features(audio_path):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "The audio file is empty."
        )

    # --------------------------------------------------------
    # Trim silence
    # --------------------------------------------------------

    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(y)
    )

    if max_value > 0:
        y = y / max_value

    # --------------------------------------------------------
    # Pad / truncate to 5 seconds
    # --------------------------------------------------------

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

    # ========================================================
    # MFCC
    # ========================================================

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

    # ========================================================
    # MEL SPECTROGRAM
    # ========================================================

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

    # ========================================================
    # CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================

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

    # ========================================================
    # EXACT FEATURE ORDER
    #
    # 40 MFCC
    # 40 MFCC delta
    # 40 MFCC delta2
    # 64 Mel
    # 12 Chroma
    # 1 centroid
    # 1 bandwidth
    # 1 rolloff
    # 1 ZCR
    # 1 RMS
    #
    # Each summarized using:
    # mean + std + min + max
    #
    # Total:
    #
    # (40 + 40 + 40 + 64 + 12 + 1 + 1 + 1 + 1 + 1) * 4
    #
    # = 201 * 4
    # = 804
    # ========================================================

    features = np.concatenate([

        summarize(mfcc),

        summarize(mfcc_delta),

        summarize(mfcc_delta2),

        summarize(mel_db),

        summarize(chroma),

        summarize(spectral_centroid),

        summarize(spectral_bandwidth),

        summarize(spectral_rolloff),

        summarize(zero_crossing_rate),

        summarize(rms)
    ])

    features = features.astype(
        np.float32
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if features.shape[0] != 804:

        raise ValueError(
            f"Expected 804 features, "
            f"got {features.shape[0]}"
        )

    if np.isnan(features).any():

        raise ValueError(
            "NaN detected in age features."
        )

    if np.isinf(features).any():

        raise ValueError(
            "Infinite value detected in age features."
        )

    return features.reshape(
        1,
        -1
    )


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(audio_path):

    features = extract_age_features(
        audio_path
    )

    # Scale exactly as during training
    features_scaled = (
        age_scaler.transform(
            features
        )
    )

    # Predict
    prediction = int(
        age_model.predict(
            features_scaled
        )[0]
    )

    age_group = AGE_CLASSES.get(
        prediction,
        "unknown"
    )

    # --------------------------------------------------------
    # Senior Citizen rule
    #
    # Dataset class is 60_plus.
    # Therefore we use 60+ as the senior threshold.
    # --------------------------------------------------------

    is_senior = (
        age_group == "60_plus"
    )

    return {
        "class_id": prediction,
        "age_group": age_group,
        "is_senior": is_senior
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Age predictor loaded successfully."
    )

    print(
        f"Model: {AGE_MODEL_PATH}"
    )

    print(
        f"Scaler: {SCALER_PATH}"
    )

    print(
        "Expected feature dimension: 804"
    )