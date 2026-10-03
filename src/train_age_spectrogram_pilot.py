from pathlib import Path
import json
import random
import time

import librosa
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "age"
MODEL_ROOT = PROJECT_ROOT / "models" / "age"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "age_models"

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SAMPLE_PER_CLASS = 500

SAMPLE_RATE = 16000
AUDIO_SECONDS = 5
AUDIO_LENGTH = SAMPLE_RATE * AUDIO_SECONDS

N_MELS = 96
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 32
EPOCHS = 20

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "00_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_plus",
]

NUM_CLASSES = len(CLASSES)


# ============================================================
# AUDIO → LOG-MEL
# ============================================================

def audio_to_logmel(path):
    """
    Load audio and convert it into a fixed-size log-Mel spectrogram.
    """

    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove silence
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # Normalize
    peak = np.max(np.abs(y))

    if peak > 0:
        y = y / peak

    # Fixed 5-second length
    if len(y) < AUDIO_LENGTH:
        y = np.pad(
            y,
            (0, AUDIO_LENGTH - len(y))
        )
    else:
        y = y[:AUDIO_LENGTH]

    # Log-Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=8000
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each spectrogram
    mean = logmel.mean()
    std = logmel.std()

    if std > 1e-8:
        logmel = (logmel - mean) / std

    return logmel.astype(np.float32)


# ============================================================
# COLLECT PILOT FILES
# ============================================================

print("=" * 70)
print("AGE SPECTROGRAM PILOT")
print("=" * 70)

train_root = DATA_ROOT / "train"
dev_root = DATA_ROOT / "dev"

rng = random.Random(SEED)


def collect_files(root, limit_per_class):
    selected = []

    print("\nCollecting files from:")
    print(root)

    for class_id, class_name in enumerate(CLASSES):

        class_dir = root / class_name

        files = list(class_dir.glob("*.mp3"))

        if len(files) == 0:
            files = list(class_dir.glob("*.wav"))

        rng.shuffle(files)

        if limit_per_class is not None:
            files = files[:limit_per_class]

        print(
            f"  {class_name:8s}: "
            f"{len(files)} files"
        )

        for path in files:
            selected.append(
                (path, class_id)
            )

    return selected


train_files = collect_files(
    train_root,
    SAMPLE_PER_CLASS
)

print("\nTraining pilot files:", len(train_files))


# ============================================================
# DEV SET
# ============================================================

# Use the complete DEV set.
dev_files = collect_files(
    dev_root,
    None
)

print("DEV files:", len(dev_files))


# ============================================================
# EXTRACT SPECTROGRAMS
# ============================================================

def build_dataset(file_list, name):

    print("\n" + "=" * 70)
    print(f"EXTRACTING {name} SPECTROGRAMS")
    print("=" * 70)

    X = []
    y = []

    failed = []

    start = time.time()

    total = len(file_list)

    for i, (path, label) in enumerate(file_list, start=1):

        try:
            feature = audio_to_logmel(path)

            X.append(feature)
            y.append(label)

        except Exception as e:

            failed.append(
                {
                    "path": str(path),
                    "error": str(e)
                }
            )

        if i % 100 == 0 or i == total:
            elapsed = time.time() - start

            print(
                f"\rProcessed "
                f"{i}/{total} "
                f"({i / total * 100:.1f}%) "
                f"| elapsed {elapsed / 60:.1f} min",
                end=""
            )

    print()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"{name} shape: {X.shape}")
    print(f"{name} labels: {y.shape}")
    print(f"Failed: {len(failed)}")

    return X, y


X_train, y_train = build_dataset(
    train_files,
    "TRAIN"
)

X_dev, y_dev = build_dataset(
    dev_files,
    "DEV"
)


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

X_train = X_train[..., np.newaxis]
X_dev = X_dev[..., np.newaxis]

print("\nFinal CNN shapes:")
print("Train:", X_train.shape)
print("Dev:  ", X_dev.shape)


# ============================================================
# CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING SPECTROGRAM CNN")
print("=" * 70)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=X_train.shape[1:]
    ),

    # Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),
    tf.keras.layers.Dropout(0.20),

    # Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),
    tf.keras.layers.Dropout(0.25),

    # Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),
    tf.keras.layers.Dropout(0.30),

    # Global representation
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.40),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

total = len(y_train)

class_weights = {}

for class_id in range(NUM_CLASSES):

    if class_counts[class_id] > 0:

        class_weights[class_id] = (
            total /
            (NUM_CLASSES * class_counts[class_id])
        )

print("\nClass weights:")

for i, weight in class_weights.items():
    print(
        f"  {CLASSES[i]}: "
        f"{weight:.4f}"
    )


# ============================================================
# CALLBACKS
# ============================================================

model_path = (
    MODEL_ROOT /
    "age_spectrogram_pilot_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SPECTROGRAM CNN PILOT")
print("=" * 70)

start = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_dev,
        y_dev
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start

print(
    f"\nTraining time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# DEV EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("SPECTROGRAM CNN PILOT — DEV RESULTS")
print("=" * 70)

probabilities = best_model.predict(
    X_dev,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


accuracy = accuracy_score(
    y_dev,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_dev,
    y_pred
)

macro_f1 = f1_score(
    y_dev,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_dev,
    y_pred,
    average="weighted"
)


print(
    f"\nAccuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_dev,
    y_pred,
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_dev,
    y_pred
)

print("Confusion Matrix:")
print(cm)

cm_path = (
    OUTPUT_ROOT /
    "age_spectrogram_pilot_confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)

print(
    "\nConfusion matrix saved:"
)

print(cm_path)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "spectrogram_2d_cnn_pilot",
    "sample_per_class": SAMPLE_PER_CLASS,
    "sample_rate": SAMPLE_RATE,
    "audio_seconds": AUDIO_SECONDS,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "training_time_minutes": float(
        training_time / 60
    ),
    "classes": CLASSES
}

results_path = (
    OUTPUT_ROOT /
    "age_spectrogram_pilot_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print(
    "\nResults saved:"
)

print(results_path)

print("\n" + "=" * 70)
print("SPECTROGRAM PILOT COMPLETE")
print("=" * 70)