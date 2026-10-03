from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING EMOTION DATA")
print("=" * 60)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    DATA_DIR / "emotion_classes.json",
    "r",
    encoding="utf-8"
) as f:

    emotion_to_index = json.load(f)

# Convert JSON mapping back to ordered list
emotion_classes = [
    emotion
    for emotion, index
    in sorted(
        emotion_to_index.items(),
        key=lambda item: item[1]
    )
]

NUM_CLASSES = len(emotion_classes)

print()
print("Emotion classes:")
for index, emotion in enumerate(emotion_classes):
    print(index, "->", emotion)

print()
print("Number of classes:", NUM_CLASSES)


# ============================================================
# CHECK DATA
# ============================================================

assert X_train.ndim == 4
assert X_val.ndim == 4
assert X_test.ndim == 4

assert X_train.shape[1:] == X_val.shape[1:]
assert X_train.shape[1:] == X_test.shape[1:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print()
print("Class weights:")
print(class_weights)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomTranslation(
            height_factor=0.04,
            width_factor=0.04,
            fill_mode="nearest"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08
        ),
    ],
    name="spectrogram_augmentation"
)


# ============================================================
# CNN MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=X_train.shape[1:],
    name="log_mel_input"
)

x = data_augmentation(inputs)

# ------------------------------------------------------------
# Block 1
# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.20
)(x)


# ------------------------------------------------------------
# Block 2
# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.25
)(x)


# ------------------------------------------------------------
# Block 3
# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)


# ------------------------------------------------------------
# Classification head
# ------------------------------------------------------------

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(
    0.40
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="emotion_output"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="RAVDESS_Male_Emotion_CNN"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR / "emotion_model_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR / "emotion_model_final.keras"
)

model.save(final_model_path)

print()
print("Final model saved:")
print(final_model_path)

print()
print("Best model saved:")
print(best_model_path)


# ============================================================
# TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Emotion Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

accuracy_plot = (
    OUTPUT_DIR / "emotion_training_accuracy.png"
)

plt.savefig(
    accuracy_plot,
    dpi=150
)

plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Emotion Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

loss_plot = (
    OUTPUT_DIR / "emotion_training_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=150
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATING ON UNSEEN TEST SPEAKERS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print()
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=emotion_classes,
    digits=4
)

print(report)


with open(
    OUTPUT_DIR / "emotion_classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("RAVDESS Male Emotion Confusion Matrix")

plt.xlabel("Predicted Emotion")

plt.ylabel("Actual Emotion")

plt.xticks(
    range(NUM_CLASSES),
    emotion_classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    emotion_classes
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_path = (
    OUTPUT_DIR / "emotion_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("EMOTION MODEL TRAINING COMPLETE")
print("=" * 60)

print()
print("Best model:")
print(best_model_path)

print()
print("Final model:")
print(final_model_path)

print()
print("Test accuracy:")
print(f"{test_accuracy:.4f}")

print()
print("Output directory:")
print(OUTPUT_DIR)