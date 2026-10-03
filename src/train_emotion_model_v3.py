import os
import json
import random
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "emotion_v3"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

GRAPH_DIR = os.path.join(
    OUTPUT_DIR,
    "graphs"
)

CM_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)

os.makedirs(
    CM_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16

EPOCHS = 60

LEARNING_RATE = 0.0003

NUM_CLASSES = 8

INPUT_SHAPE = (
    128,
    251,
    3
)

CLASS_NAMES = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EMOTION MODEL V3 TRAINING")
print("=" * 70)

print()
print("Loading V3 datasets...")


X_train = np.load(
    os.path.join(
        DATA_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "y_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        DATA_DIR,
        "X_validation.npy"
    )
)

y_val = np.load(
    os.path.join(
        DATA_DIR,
        "y_validation.npy"
    )
)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test.npy"
    )
)


# ============================================================
# DATA INFORMATION
# ============================================================

print()
print("Dataset shapes:")

print(
    f"X_train: {X_train.shape}"
)

print(
    f"y_train: {y_train.shape}"
)

print(
    f"X_val:   {X_val.shape}"
)

print(
    f"y_val:   {y_val.shape}"
)

print(
    f"X_test:  {X_test.shape}"
)

print(
    f"y_test:  {y_test.shape}"
)


# ============================================================
# VALIDATE INPUT
# ============================================================

if X_train.shape[1:] != INPUT_SHAPE:

    raise ValueError(
        f"Unexpected training shape: "
        f"{X_train.shape[1:]}"
    )

if X_val.shape[1:] != INPUT_SHAPE:

    raise ValueError(
        f"Unexpected validation shape: "
        f"{X_val.shape[1:]}"
    )

if X_test.shape[1:] != INPUT_SHAPE:

    raise ValueError(
        f"Unexpected test shape: "
        f"{X_test.shape[1:]}"
    )


# ============================================================
# CHECK FOR NaN / INF
# ============================================================

print()
print("Checking data quality...")

print(
    "Train NaN:",
    np.isnan(X_train).sum()
)

print(
    "Train Inf:",
    np.isinf(X_train).sum()
)

print(
    "Validation NaN:",
    np.isnan(X_val).sum()
)

print(
    "Validation Inf:",
    np.isinf(X_val).sum()
)

print(
    "Test NaN:",
    np.isnan(X_test).sum()
)

print(
    "Test Inf:",
    np.isinf(X_test).sum()
)


if (
    np.isnan(X_train).any()
    or np.isinf(X_train).any()
    or np.isnan(X_val).any()
    or np.isinf(X_val).any()
    or np.isnan(X_test).any()
    or np.isinf(X_test).any()
):

    raise ValueError(
        "Dataset contains NaN or Inf values."
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

for label in range(NUM_CLASSES):

    train_count = np.sum(
        y_train == label
    )

    val_count = np.sum(
        y_val == label
    )

    test_count = np.sum(
        y_test == label
    )

    print(
        f"{label}: "
        f"{CLASS_NAMES[label]:<10} "
        f"Train={train_count:<3} "
        f"Val={val_count:<3} "
        f"Test={test_count:<3}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    y_train
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        weights
    )
}

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

for label in range(NUM_CLASSES):

    print(
        f"{label}: "
        f"{CLASS_NAMES[label]:<10} "
        f"{class_weights[label]:.4f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        # Small frequency/time translations
        tf.keras.layers.RandomTranslation(
            height_factor=0.04,
            width_factor=0.04,
            fill_mode="reflect"
        ),

        # Small zoom
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

def build_model():

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="input_spectrogram"
    )

    x = data_augmentation(
        inputs
    )

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)


    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)


    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)


    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # DENSE LAYER
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        128,
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(
        "relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.40
    )(x)


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Emotion_CNN_V3"
    )

    return model


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = build_model()

model.summary()


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "emotion_model_v3_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "emotion_model_v3_final.keras"
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=12,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(
        OUTPUT_DIR,
        "emotion_v3_training_log.csv"
    )
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint,
        csv_logger
    ],

    shuffle=True,

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)

print()
print(
    f"Final model saved to:\n"
    f"{final_model_path}"
)

print(
    f"Best model saved to:\n"
    f"{best_model_path}"
)


# ============================================================
# TRAINING CURVES
# ============================================================

print()
print(
    "Creating training graphs..."
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Emotion Model V3 Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "emotion_v3_accuracy.png"
    ),
    dpi=150
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Emotion Model V3 Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "emotion_v3_loss.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print(
    "Loading best model..."
)

best_model = tf.keras.models.load_model(
    best_model_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = (
    best_model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )
)

print()
print(
    f"Test Loss: {test_loss:.6f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.6f}"
)

print(
    f"Test Accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print(
    "Generating test predictions..."
)

probabilities = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for label in range(NUM_CLASSES):

    count = np.sum(
        predictions == label
    )

    percentage = (
        count / len(predictions)
    ) * 100

    print(
        f"{label}: "
        f"{CLASS_NAMES[label]:<10} "
        f"{count:<3} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    predictions,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(
    report
)

report_path = os.path.join(
    OUTPUT_DIR,
    "emotion_v3_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(
        range(NUM_CLASSES)
    )
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "True Emotion"
)

plt.title(
    "Emotion Model V3 Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    CM_DIR,
    "emotion_v3_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_data = {

    "true_labels":
        y_test.tolist(),

    "predicted_labels":
        predictions.tolist(),

    "predicted_emotions": [
        CLASS_NAMES[int(x)]
        for x in predictions
    ],

    "confidence": [
        float(np.max(p))
        for p in probabilities
    ]
}

prediction_path = os.path.join(
    OUTPUT_DIR,
    "emotion_v3_predictions.json"
)

with open(
    prediction_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        prediction_data,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EMOTION MODEL V3 COMPLETE")
print("=" * 70)

print()
print(
    f"Best model:\n"
    f"{best_model_path}"
)

print()
print(
    f"Final model:\n"
    f"{final_model_path}"
)

print()
print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print(
    "Training graph:"
)

print(
    os.path.join(
        GRAPH_DIR,
        "emotion_v3_accuracy.png"
    )
)

print()
print(
    "Confusion matrix:"
)

print(
    cm_path
)

print()
print("=" * 70)