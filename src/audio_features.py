from pathlib import Path

import librosa
import numpy as np


# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 16000

# We will use a fixed 4-second input.
# RAVDESS recordings are around 3–4 seconds.
DURATION = 4

TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MELS = 128

N_FFT = 1024

HOP_LENGTH = 256


# ============================================================
# Load and standardize audio
# ============================================================

def load_audio(
    file_path,
    sample_rate=SAMPLE_RATE,
    target_length=TARGET_LENGTH
):
    """
    Load an audio file and standardize it.

    Output:
        1D numpy array with fixed length.
    """

    audio, _ = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize amplitude
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Fix duration
    if len(audio) < target_length:

        padding = target_length - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    else:

        audio = audio[:target_length]

    return audio.astype(np.float32)


# ============================================================
# Log-Mel Spectrogram
# ============================================================

def audio_to_log_mel(
    file_path,
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH
):
    """
    Convert audio into a normalized Log-Mel Spectrogram.

    Output:
        numpy array with shape:
        (n_mels, time_frames)
    """

    audio = load_audio(
        file_path,
        sample_rate=sample_rate
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sample_rate // 2,
        power=2.0
    )

    # Convert power spectrogram to decibels
    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize approximately to 0–1
    log_mel = (log_mel + 80.0) / 80.0

    log_mel = np.clip(
        log_mel,
        0.0,
        1.0
    )

    return log_mel.astype(np.float32)