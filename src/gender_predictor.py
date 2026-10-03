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
    / "gender"
)

GENDER_MODEL_PATH = (
    MODEL_ROOT
    / "gender_model.joblib"
)

SCALER_PATH = (
    MODEL_ROOT
    / "gender_scaler.joblib"
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
# LOAD MODEL
# ============================================================

gender_model = joblib.load(
    GENDER_MODEL_PATH
)

gender_scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_gender_features(
    audio_path
):

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "The audio file is empty."
        )

    # Remove silence
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # Normalize
    max_value = np.max(
        np.abs(y)
    )

    if max_value > 0:
        y = y / max_value

    # Pad / truncate
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

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mel
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Summary function
    # --------------------------------------------------------

    def summarize(feature):

        return np.concatenate([
            np.mean(feature, axis=1),
            np.std(feature, axis=1),
            np.min(feature, axis=1),
            np.max(feature, axis=1)
        ])

    # --------------------------------------------------------
    # Final 1320-dimensional vector
    # --------------------------------------------------------

    features = np.concatenate([

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

    features = features.astype(
        np.float32
    )

    if np.isnan(features).any():
        raise ValueError(
            "NaN detected in audio features."
        )

    if np.isinf(features).any():
        raise ValueError(
            "Infinite value detected in audio features."
        )

    return features.reshape(
        1,
        -1
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_gender(
    audio_path
):

    features = extract_gender_features(
        audio_path
    )

    features_scaled = (
        gender_scaler.transform(
            features
        )
    )

    prediction = int(
        gender_model.predict(
            features_scaled
        )[0]
    )

    gender = (
        "male"
        if prediction == 1
        else "female"
    )

    confidence = None

    # LogisticRegression supports predict_proba
    if hasattr(
        gender_model,
        "predict_proba"
    ):

        probabilities = (
            gender_model.predict_proba(
                features_scaled
            )[0]
        )

        confidence = float(
            np.max(probabilities)
        )

    return {
        "gender": gender,
        "confidence": confidence
    }


# ============================================================
# TEST FROM COMMAND LINE
# ============================================================

if __name__ == "__main__":

    print(
        "Gender predictor loaded successfully."
    )

    print(
        f"Model: {GENDER_MODEL_PATH}"
    )