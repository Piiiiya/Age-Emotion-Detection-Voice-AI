from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
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

FEATURE_ROOT = PROJECT_ROOT / "data" / "processed" / "age_features"
MODEL_ROOT = PROJECT_ROOT / "models" / "age"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "age_models"

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGE CNN TRAINING")
print("=" * 70)

print("\nLoading feature matrices...")

X_train = np.load(FEATURE_ROOT / "train_features.npy").astype("float32")
y_train = np.load(FEATURE_ROOT / "train_labels.npy")

X_dev = np.load(FEATURE_ROOT / "dev_features.npy").astype("float32")
y_dev = np.load(FEATURE_ROOT / "dev_labels.npy")

print(f"Train: {X_train.shape}")
print(f"Dev:   {X_dev.shape}")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(FEATURE_ROOT / "age_classes.json", "r", encoding="utf-8") as f:
    class_info = json.load(f)

if isinstance(class_info, dict):
    classes = [
        name for name, index
        in sorted(class_info.items(), key=lambda x: x[1])
    ]
else:
    classes = list(class_info)

NUM_CLASSES = len(classes)

print("\nClasses:")
for i, name in enumerate(classes):
    print(f"  {i}: {name}")

# ============================================================
# NORMALIZE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING FEATURES")
print("=" * 70)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std < 1e-8] = 1.0

X_train = (X_train - mean) / std
X_dev = (X_dev - mean) / std

# Save normalization parameters
np.save(MODEL_ROOT / "age_cnn_mean.npy", mean)
np.save(MODEL_ROOT / "age_cnn_std.npy", std)

print("Normalization completed.")

# ============================================================
# RESHAPE FOR 1D CNN
# ============================================================

X_train = X_train[..., np.newaxis]
X_dev = X_dev[..., np.newaxis]

print(f"\nCNN train shape: {X_train.shape}")
print(f"CNN dev shape:   {X_dev.shape}")

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        np.unique(y_train),
        class_weights_array
    )
}

print("\nClass weights:")

for class_id, weight in class_weights.items():
    print(f"  {classes[class_id]}: {weight:.4f}")

# ============================================================
# CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CNN")
print("=" * 70)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(804, 1)),

    tf.keras.layers.Conv1D(
        64,
        kernel_size=7,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv1D(
        128,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Conv1D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.35),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        256,
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
# CALLBACKS
# ============================================================

model_path = MODEL_ROOT / "age_cnn_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN")
print("=" * 70)

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_dev, y_dev),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

print(f"\nTraining time: {training_time / 60:.2f} minutes")

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best CNN model...")

best_model = tf.keras.models.load_model(model_path)

# ============================================================
# DEV PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("AGE CNN — DEV RESULTS")
print("=" * 70)

probabilities = best_model.predict(
    X_dev,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_dev, y_pred)

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

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report_text = classification_report(
    y_dev,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report_text)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_dev,
    y_pred
)

print("Confusion Matrix:")
print(cm)

cm_path = OUTPUT_ROOT / "age_cnn_dev_confusion_matrix.npy"

np.save(
    cm_path,
    cm
)

print(f"\nConfusion matrix saved:")
print(cm_path)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "1D_CNN",
    "input_features": 804,
    "num_classes": NUM_CLASSES,
    "classes": classes,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "training_time_minutes": training_time / 60,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
}

results_path = OUTPUT_ROOT / "age_cnn_dev_results.json"

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

print("\nResults saved:")
print(results_path)

print("\n" + "=" * 70)
print("AGE CNN TRAINING COMPLETE")
print("=" * 70)