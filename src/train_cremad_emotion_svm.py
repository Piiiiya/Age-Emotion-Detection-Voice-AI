# ============================================================
# CREMA-D EMOTION MODEL V3
# HANDCRAFTED AUDIO FEATURES + SVM
#
# Speaker-independent split
# Male voices only
# 6 emotion classes
#
# Classes:
# 0 = angry
# 1 = disgust
# 2 = fearful
# 3 = happy
# 4 = neutral
# 5 = sad
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import librosa
import joblib

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print("=" * 70)
print("CREMA-D EMOTION MODEL V3")
print("HANDCRAFTED AUDIO FEATURES + SVM")
print("=" * 70)

print("\nProject directory:")
print(BASE_DIR)


# ============================================================
# 2. CREMA-D AUDIO DIRECTORY
# ============================================================

CREMAD_AUDIO_DIR = os.path.join(
    BASE_DIR,
    "crema_d",
    "AudioWAV"
)

print("\nExpected CREMA-D audio directory:")
print(CREMAD_AUDIO_DIR)


# ============================================================
# 3. METADATA DIRECTORY
# ============================================================

METADATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "metadata",
    "cremad_splits"
)


# ============================================================
# 4. OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 5. RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# 6. AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

DURATION = 4.0

TARGET_LENGTH = int(
    SAMPLE_RATE * DURATION
)


# ============================================================
# 7. EMOTION CLASSES
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
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# 8. CHECK CREMA-D AUDIO DIRECTORY
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING CREMA-D AUDIO DIRECTORY")
print("=" * 70)

if not os.path.exists(
    CREMAD_AUDIO_DIR
):

    raise FileNotFoundError(
        "\nCREMA-D AudioWAV folder was NOT found.\n\n"
        f"Expected location:\n"
        f"{CREMAD_AUDIO_DIR}\n\n"
        "Please check the CREMA-D repository location."
    )


if not os.path.isdir(
    CREMAD_AUDIO_DIR
):

    raise NotADirectoryError(
        "\nThe CREMA-D AudioWAV path exists "
        "but is not a directory:\n"
        f"{CREMAD_AUDIO_DIR}"
    )


print(
    "Audio directory found successfully."
)


# ============================================================
# 9. INDEX ALL WAV FILES
# ============================================================

print("\n")
print("=" * 70)
print("INDEXING CREMA-D AUDIO FILES")
print("=" * 70)

audio_index = {}

wav_count = 0


for root, dirs, files in os.walk(
    CREMAD_AUDIO_DIR
):

    for filename in files:

        if filename.lower().endswith(
            ".wav"
        ):

            wav_count += 1

            full_path = os.path.join(
                root,
                filename
            )

            audio_index[
                filename.lower()
            ] = full_path


print(
    "\nCREMA-D audio directory:"
)

print(
    CREMAD_AUDIO_DIR
)

print(
    "\nWAV files found:",
    wav_count
)


if wav_count == 0:

    raise RuntimeError(
        "\nNo WAV files were found inside:\n"
        f"{CREMAD_AUDIO_DIR}\n\n"
        "Your CREMA-D audio dataset appears "
        "to be missing."
    )


print(
    "\nFirst 10 WAV files found:"
)


for index, path in enumerate(
    list(audio_index.values())[:10]
):

    print(
        f"{index + 1}. {path}"
    )


# ============================================================
# 10. FIND AUDIO FILE
# ============================================================

def find_audio_file(filename):

    """
    Find a CREMA-D WAV file by filename.

    Matching is case-insensitive and works
    regardless of subfolder placement.
    """

    if pd.isna(filename):

        return None


    filename = os.path.basename(
        str(filename).strip()
    )


    return audio_index.get(
        filename.lower()
    )


# ============================================================
# 11. VERIFY METADATA FILES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING METADATA FILES")
print("=" * 70)


TRAIN_CSV = os.path.join(
    METADATA_DIR,
    "cremad_male_train.csv"
)

VAL_CSV = os.path.join(
    METADATA_DIR,
    "cremad_male_val.csv"
)

TEST_CSV = os.path.join(
    METADATA_DIR,
    "cremad_male_test.csv"
)


for csv_path in [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV
]:

    if not os.path.exists(
        csv_path
    ):

        raise FileNotFoundError(
            "\nMetadata file not found:\n"
            f"{csv_path}"
        )

    print(
        "Found:",
        csv_path
    )


# ============================================================
# 12. LOAD AUDIO
# ============================================================

def load_audio(path):

    """
    Load audio as mono at 16 kHz
    and make every recording exactly 4 seconds.
    """

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )


    # Remove DC offset

    audio = (
        audio -
        np.mean(audio)
    )


    # Normalize amplitude

    max_value = np.max(
        np.abs(audio)
    )


    if max_value > 0:

        audio = (
            audio /
            max_value
        )


    # Pad or crop to exactly 4 seconds

    if len(audio) < TARGET_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                TARGET_LENGTH - len(audio)
            )
        )

    elif len(audio) > TARGET_LENGTH:

        audio = audio[
            :TARGET_LENGTH
        ]


    return audio


# ============================================================
# 13. FEATURE STATISTICS
# ============================================================

def statistics(feature):

    """
    Convert a time-series feature into
    fixed-size statistical features.

    Mean
    Standard deviation
    Minimum
    Maximum
    """

    feature = np.asarray(
        feature,
        dtype=np.float32
    )


    mean_value = np.mean(
        feature,
        axis=1
    )

    std_value = np.std(
        feature,
        axis=1
    )

    min_value = np.min(
        feature,
        axis=1
    )

    max_value = np.max(
        feature,
        axis=1
    )


    return np.concatenate(
        [
            mean_value,
            std_value,
            min_value,
            max_value
        ]
    )


# ============================================================
# 14. EXTRACT AUDIO FEATURES
# ============================================================

def extract_features(path):

    """
    Extract handcrafted speech/audio features.
    """

    audio = load_audio(
        path
    )


    features = []


    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(mfcc)
    )


    # ========================================================
    # MFCC DELTA
    # ========================================================

    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    features.extend(
        statistics(mfcc_delta)
    )


    # ========================================================
    # MFCC DELTA-DELTA
    # ========================================================

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(
        statistics(mfcc_delta2)
    )


    # ========================================================
    # MEL SPECTROGRAM
    # ========================================================

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=20,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    features.extend(
        statistics(mel_db)
    )


    # ========================================================
    # CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(chroma)
    )


    # ========================================================
    # SPECTRAL CONTRAST
    # ========================================================

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(contrast)
    )


    # ========================================================
    # ZERO CROSSING RATE
    # ========================================================

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=1024,
        hop_length=256
    )

    features.extend(
        statistics(zcr)
    )


    # ========================================================
    # RMS ENERGY
    # ========================================================

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256
    )

    features.extend(
        statistics(rms)
    )


    # ========================================================
    # SPECTRAL CENTROID
    # ========================================================

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(centroid)
    )


    # ========================================================
    # SPECTRAL BANDWIDTH
    # ========================================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(bandwidth)
    )


    # ========================================================
    # SPECTRAL ROLLOFF
    # ========================================================

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256
    )

    features.extend(
        statistics(rolloff)
    )


    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    feature_vector = np.asarray(
        features,
        dtype=np.float32
    )


    return feature_vector


# ============================================================
# 15. PROCESS DATASET
# ============================================================

def process_dataset(
    csv_path,
    dataset_name
):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING {dataset_name.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )


    print(
        "\nMetadata recordings:",
        len(df)
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "file",
        "emotion"
    ]


    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"\nRequired column '{column}' "
                f"was not found in:\n"
                f"{csv_path}\n\n"
                f"Available columns:\n"
                f"{list(df.columns)}"
            )


    # --------------------------------------------------------
    # Feature storage
    # --------------------------------------------------------

    X = []

    y = []

    failed = []


    # --------------------------------------------------------
    # Process audio
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=dataset_name
    ):

        filename = row["file"]

        emotion = str(
            row["emotion"]
        ).strip().lower()


        # ----------------------------------------------------
        # Check emotion
        # ----------------------------------------------------

        if emotion not in CLASS_TO_ID:

            failed.append(
                (
                    filename,
                    f"Unknown emotion: {emotion}"
                )
            )

            continue


        # ----------------------------------------------------
        # Find WAV
        # ----------------------------------------------------

        audio_path = find_audio_file(
            filename
        )


        if audio_path is None:

            failed.append(
                (
                    filename,
                    "Audio file not found"
                )
            )

            continue


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        try:

            feature_vector = extract_features(
                audio_path
            )


            if feature_vector.ndim != 1:

                raise ValueError(
                    "Feature vector is not 1-dimensional."
                )


            if not np.all(
                np.isfinite(
                    feature_vector
                )
            ):

                raise ValueError(
                    "Feature vector contains NaN or infinity."
                )


            X.append(
                feature_vector
            )


            y.append(
                CLASS_TO_ID[emotion]
            )


        except Exception as error:

            failed.append(
                (
                    filename,
                    str(error)
                )
            )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(X) == 0:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ERROR: ZERO AUDIO FEATURES EXTRACTED"
        )

        print(
            "=" * 70
        )


        print(
            "\nExpected recordings:",
            len(df)
        )

        print(
            "Successful:",
            0
        )

        print(
            "Failed:",
            len(failed)
        )


        print(
            "\nFirst 20 failures:"
        )


        for failure in failed[:20]:

            print(
                failure
            )


        raise RuntimeError(
            f"\nNo audio features were extracted "
            f"for {dataset_name}.\n"
            "Check the CREMA-D audio path, "
            "filenames, and audio files."
        )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X.ndim != 2:

        raise ValueError(
            f"\n{dataset_name} feature array "
            f"must be 2D but got:\n"
            f"{X.shape}"
        )


    if len(X) != len(y):

        raise ValueError(
            f"\nFeature/label mismatch:\n"
            f"X = {len(X)}\n"
            f"y = {len(y)}"
        )


    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"{dataset_name} feature shape:",
        X.shape
    )

    print(
        f"{dataset_name} label shape:",
        y.shape
    )

    print(
        "Successful:",
        len(X)
    )

    print(
        "Failed:",
        len(failed)
    )


    print(
        "\nEmotion distribution:"
    )


    unique_classes, class_counts = np.unique(
        y,
        return_counts=True
    )


    for class_id, count in zip(
        unique_classes,
        class_counts
    ):

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:10s} "
            f"-> {count}"
        )


    # --------------------------------------------------------
    # Save failure report if necessary
    # --------------------------------------------------------

    if failed:

        failure_path = os.path.join(
            OUTPUT_DIR,
            f"{dataset_name}_failed.txt"
        )


        with open(
            failure_path,
            "w",
            encoding="utf-8"
        ) as f:

            for filename, reason in failed:

                f.write(
                    f"{filename}\t{reason}\n"
                )


    return X, y, failed


# ============================================================
# 16. PROCESS TRAINING DATA
# ============================================================

X_train, y_train, failed_train = process_dataset(
    TRAIN_CSV,
    "train"
)


# ============================================================
# 17. PROCESS VALIDATION DATA
# ============================================================

X_val, y_val, failed_val = process_dataset(
    VAL_CSV,
    "validation"
)


# ============================================================
# 18. PROCESS TEST DATA
# ============================================================

X_test, y_test, failed_test = process_dataset(
    TEST_CSV,
    "test"
)


# ============================================================
# 19. SAVE EXTRACTED FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("SAVING FEATURE ARRAYS")
print("=" * 70)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


print(
    "Feature arrays saved."
)


# ============================================================
# 20. BUILD SVM
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING SVM")
print("=" * 70)


model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            )
        )
    ]
)


print(
    "\nSVM configuration:"
)

print(
    "Kernel: RBF"
)

print(
    "C: 10"
)

print(
    "Gamma: scale"
)

print(
    "Class weight: balanced"
)


# ============================================================
# 21. TRAIN SVM
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING SVM")
print("=" * 70)


print(
    "\nTraining samples:",
    X_train.shape[0]
)

print(
    "Features per sample:",
    X_train.shape[1]
)


model.fit(
    X_train,
    y_train
)


print(
    "\nSVM training complete."
)


# ============================================================
# 22. VALIDATION PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


val_pred = model.predict(
    X_val
)


val_accuracy = accuracy_score(
    y_val,
    val_pred
)


val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    val_pred
)


print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)


print(
    "Validation Balanced Accuracy:",
    f"{val_balanced_accuracy * 100:.2f}%"
)


# ============================================================
# 23. TEST PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_pred = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_pred
)


test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_pred
)


print(
    "\nTest Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


print(
    "Test Balanced Accuracy:",
    f"{test_balanced_accuracy * 100:.2f}%"
)


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    test_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    report
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


print(
    cm
)


# ============================================================
# 26. PREDICTION DISTRIBUTION
# ============================================================

unique_predictions, prediction_counts = np.unique(
    test_pred,
    return_counts=True
)


print("\n")
print("=" * 70)
print("TEST PREDICTION DISTRIBUTION")
print("=" * 70)


for class_id, count in zip(
    unique_predictions,
    prediction_counts
):

    percentage = (
        count /
        len(test_pred)
    ) * 100


    print(
        f"{CLASS_NAMES[class_id]:10s}: "
        f"{count:3d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 27. SAVE SVM MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_svm.joblib"
)


joblib.dump(
    model,
    model_path
)


print(
    "\nSVM model saved:"
)

print(
    model_path
)


# ============================================================
# 28. SAVE CLASS MAPPING
# ============================================================

class_mapping_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_svm_classes.json"
)


with open(
    class_mapping_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            str(index): name
            for index, name
            in enumerate(CLASS_NAMES)
        },
        f,
        indent=4
    )


# ============================================================
# 29. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CREMA-D EMOTION MODEL V3\n"
    )

    f.write(
        "HANDCRAFTED AUDIO FEATURES + SVM\n\n"
    )

    f.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_val)}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(X_test)}\n\n"
    )

    f.write(
        f"Number of features: "
        f"{X_train.shape[1]}\n\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Validation Balanced Accuracy: "
        f"{val_balanced_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report:\n\n"
    )

    f.write(
        report
    )


# ============================================================
# 30. SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.txt"
)


np.savetxt(
    cm_path,
    cm,
    fmt="%d"
)


# ============================================================
# 31. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "model_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CREMA-D Emotion Model V3\n"
    )

    f.write(
        "Handcrafted Audio Features + RBF SVM\n\n"
    )

    f.write(
        f"Audio files indexed: "
        f"{wav_count}\n"
    )

    f.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_val)}\n"
    )

    f.write(
        f"Test samples: "
        f"{len(X_test)}\n"
    )

    f.write(
        f"Feature count: "
        f"{X_train.shape[1]}\n\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Validation Balanced Accuracy: "
        f"{val_balanced_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%\n"
    )


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("CREMA-D EMOTION MODEL V3 COMPLETE")
print("=" * 70)


print(
    "\nAudio files indexed:",
    wav_count
)


print(
    "Training samples:",
    len(X_train)
)


print(
    "Validation samples:",
    len(X_val)
)


print(
    "Test samples:",
    len(X_test)
)


print(
    "Features per sample:",
    X_train.shape[1]
)


print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)


print(
    "Validation Balanced Accuracy:",
    f"{val_balanced_accuracy * 100:.2f}%"
)


print(
    "\nTest Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


print(
    "Test Balanced Accuracy:",
    f"{test_balanced_accuracy * 100:.2f}%"
)


print(
    "\nModel:",
    model_path
)


print(
    "\nClassification report:",
    report_path
)


print(
    "\nConfusion matrix:",
    cm_path
)


print(
    "\n"
    + "=" * 70
)

print(
    "ALL PROCESSING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)