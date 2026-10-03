from pathlib import Path
import json

import joblib
import librosa
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "emotion_cremad_svm_v5.joblib"

CLASSES = {
    0: "angry",
    1: "disgust",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
}


# ============================================================
# AUDIO CONFIGURATION
# EXACTLY MATCHES CREMA-D V5 TRAINING
# ============================================================

SR = 16000
DURATION = 4
TARGET_LENGTH = SR * DURATION

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# FEATURE STATISTICS
# ============================================================

def statistics(x):
    """
    Calculate the same five statistics used during V5 training.

    Output:
        mean
        std
        min
        max
        median
    """

    x = np.asarray(x)

    return np.concatenate(
        [
            np.mean(x, axis=1),
            np.std(x, axis=1),
            np.min(x, axis=1),
            np.max(x, axis=1),
            np.median(x, axis=1),
        ]
    )


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):
    """
    Load audio as mono 16 kHz and make it exactly 4 seconds.
    """

    audio, _ = librosa.load(
        str(path),
        sr=SR,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Remove leading/trailing silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Exactly 4 seconds
    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (0, TARGET_LENGTH - len(audio)),
            mode="constant"
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

    features.append(statistics(mfcc))

    # MFCC delta
    mfcc_delta = librosa.feature.delta(mfcc)

    features.append(statistics(mfcc_delta))

    # MFCC delta-delta
    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.append(statistics(mfcc_delta2))

    # --------------------------------------------------------
    # 2. MEL SPECTROGRAM
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

    features.append(statistics(mel_db))

    # Mel delta
    mel_delta = librosa.feature.delta(mel_db)

    features.append(statistics(mel_delta))

    # Mel delta-delta
    mel_delta2 = librosa.feature.delta(
        mel_db,
        order=2
    )

    features.append(statistics(mel_delta2))

    # --------------------------------------------------------
    # 3. CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(chroma))

    # --------------------------------------------------------
    # 4. SPECTRAL CONTRAST
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(contrast))

    # --------------------------------------------------------
    # 5. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(zcr))

    # --------------------------------------------------------
    # 6. RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(rms))

    # RMS delta
    rms_delta = librosa.feature.delta(rms)

    features.append(statistics(rms_delta))

    # RMS delta-delta
    rms_delta2 = librosa.feature.delta(
        rms,
        order=2
    )

    features.append(statistics(rms_delta2))

    # --------------------------------------------------------
    # 7. SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(centroid))

    # --------------------------------------------------------
    # 8. SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(bandwidth))

    # --------------------------------------------------------
    # 9. SPECTRAL ROLLOFF
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(rolloff))

    # --------------------------------------------------------
    # 10. SPECTRAL FLATNESS
    # --------------------------------------------------------

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.append(statistics(flatness))

    # --------------------------------------------------------
    # FINAL VECTOR
    # --------------------------------------------------------

    feature_vector = np.concatenate(features)

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    return feature_vector


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Emotion model not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    features = extract_features(audio_path)

    # Safety check
    if features.shape[0] != 1695:

        raise ValueError(
            f"Feature dimension mismatch!\n"
            f"Expected: 1695\n"
            f"Got: {features.shape[0]}"
        )

    X = features.reshape(1, -1)

    prediction = model.predict(X)[0]

    emotion = CLASSES[int(prediction)]

    result = {
        "class_id": int(prediction),
        "emotion": emotion,
        "feature_dimension": int(features.shape[0])
    }

    # If the SVM supports decision scores,
    # expose them for debugging only.
    if hasattr(model, "decision_function"):

        decision = model.decision_function(X)

        result["decision_scores"] = (
            np.asarray(decision)
            .flatten()
            .tolist()
        )

    return result


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CREMA-D EMOTION V5 PREDICTOR")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Expected features: 1695")

    print("\nEmotion classes:")

    for class_id, name in CLASSES.items():
        print(f"  {class_id}: {name}")

    print("\nPredictor loaded successfully.")
    print("=" * 60)