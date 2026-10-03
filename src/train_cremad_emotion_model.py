# ================================================================
# CREMA-D EMOTION MODEL TRAINING
# ================================================================

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "emotion_cremad"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "emotion_cremad"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# SETTINGS
# ================================================================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0003
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("CREMA-D EMOTION MODEL TRAINING")
print("=" * 70)


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading datasets...")


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
        "X_val.npy"
    )
)

y_val = np.load(
    os.path.join(
        DATA_DIR,
        "y_val.npy"
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


# ================================================================
# LOAD CLASS MAPPING
# ================================================================

classes_path = os.path.join(
    DATA_DIR,
    "classes.json"
)


with open(
    classes_path,
    "r"
) as f:

    classes_raw = json.load(f)


print("\nRaw class mapping:")

print(classes_raw)


# ================================================================
# HANDLE CLASS MAPPING
# ================================================================

# Your classes.json is expected to be:
#
# {
#     "angry": 0,
#     "disgust": 1,
#     "fearful": 2,
#     "happy": 3,
#     "neutral": 4,
#     "sad": 5
# }
#
# We convert it into:
#
# {
#     0: "angry",
#     1: "disgust",
#     2: "fearful",
#     "..."
# }


if isinstance(classes_raw, dict):

    first_key = next(
        iter(classes_raw)
    )

    first_value = classes_raw[
        first_key
    ]


    # Format:
    # {"angry": 0, "disgust": 1, ...}

    if isinstance(first_value, int):

        classes = {
            int(value): str(key)
            for key, value
            in classes_raw.items()
        }


    # Format:
    # {"0": "angry", "1": "disgust", ...}

    else:

        classes = {
            int(key): str(value)
            for key, value
            in classes_raw.items()
        }


else:

    raise ValueError(
        "Unsupported classes.json format."
    )


# ================================================================
# DATASET INFORMATION
# ================================================================

print("\nDataset shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val  :",
    X_val.shape
)

print(
    "y_val  :",
    y_val.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_test :",
    y_test.shape
)


# ================================================================
# CLASSES
# ================================================================

num_classes = len(classes)


print("\nClasses:")

for class_id in sorted(
    classes.keys()
):

    print(
        f"{class_id}: "
        f"{classes[class_id]}"
    )


print(
    "\nNumber of classes:",
    num_classes
)


# ================================================================
# TRAINING DISTRIBUTION
# ================================================================

print("\nTraining distribution:")


for class_id in range(
    num_classes
):

    count = np.sum(
        y_train == class_id
    )

    print(
        f"{class_id}: "
        f"{classes[class_id]} -> "
        f"{count}"
    )


# ================================================================
# VALIDATION DISTRIBUTION
# ================================================================

print("\nValidation distribution:")


for class_id in range(
    num_classes
):

    count = np.sum(
        y_val == class_id
    )

    print(
        f"{class_id}: "
        f"{classes[class_id]} -> "
        f"{count}"
    )


# ================================================================
# TEST DISTRIBUTION
# ================================================================

print("\nTest distribution:")


for class_id in range(
    num_classes
):

    count = np.sum(
        y_test == class_id
    )

    print(
        f"{class_id}: "
        f"{classes[class_id]} -> "
        f"{count}"
    )


# ================================================================
# CLASS WEIGHTS
# ================================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(
        num_classes
    ),
    y=y_train
)


class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")


for class_id in range(
    num_classes
):

    print(
        f"{class_id}: "
        f"{classes[class_id]} -> "
        f"{class_weights[class_id]:.4f}"
    )


# ================================================================
# DATA AUGMENTATION
# ================================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="reflect"
        ),

        layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="reflect"
        )

    ],
    name="spectrogram_augmentation"
)


# ================================================================
# INPUT
# ================================================================

input_shape = X_train.shape[1:]


print(
    "\nInput shape:",
    input_shape
)


inputs = layers.Input(
    shape=input_shape,
    name="mel_input"
)


# ================================================================
# DATA AUGMENTATION
# ================================================================

x = data_augmentation(
    inputs
)


# ================================================================
# CNN BLOCK 1
# ================================================================

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)

x = layers.Dropout(
    0.20
)(x)


# ================================================================
# CNN BLOCK 2
# ================================================================

x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)

x = layers.Dropout(
    0.25
)(x)


# ================================================================
# CNN BLOCK 3
# ================================================================

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)

x = layers.Dropout(
    0.30
)(x)


# ================================================================
# CNN BLOCK 4
# ================================================================

x = layers.Conv2D(
    256,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    256,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)

x = layers.Dropout(
    0.35
)(x)


# ================================================================
# CLASSIFICATION HEAD
# ================================================================

x = layers.GlobalAveragePooling2D()(
    x
)

x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=regularizers.l2(
        1e-4
    )
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.45
)(x)


# ================================================================
# OUTPUT
# ================================================================

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="emotion_output"
)(x)


# ================================================================
# CREATE MODEL
# ================================================================

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="CREMAD_Emotion_CNN"
)


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ================================================================
# COMPILE
# ================================================================

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ================================================================
# MODEL SUMMARY
# ================================================================

print("\n")

model.summary()


# ================================================================
# MODEL PATHS
# ================================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_final.keras"
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


# ================================================================
# CALLBACKS
# ================================================================

checkpoint = callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


csv_logger = callbacks.CSVLogger(
    history_path
)


# ================================================================
# START TRAINING
# ================================================================

print("\n")

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
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],

    verbose=1
)


# ================================================================
# SAVE FINAL MODEL
# ================================================================

model.save(
    final_model_path
)


print("\nFinal model saved:")
print(
    final_model_path
)


print("\nBest model saved:")
print(
    best_model_path
)


# ================================================================
# TEST EVALUATION
# ================================================================

print("\n")

print("=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


print(
    "\nTest Loss:",
    test_loss
)


print(
    "Test Accuracy:",
    test_accuracy
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ================================================================
# PREDICTIONS
# ================================================================

print("\nGenerating predictions...")


y_probability = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


y_pred = np.argmax(
    y_probability,
    axis=1
)


# ================================================================
# CLASS NAMES
# ================================================================

class_names = [

    classes[i]

    for i in range(
        num_classes
    )

]


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n")

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_test,

    y_pred,

    labels=np.arange(
        num_classes
    ),

    target_names=class_names,

    digits=4,

    zero_division=0

)


print(report)


report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(report)


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=np.arange(
        num_classes
    )

)


print("\nConfusion Matrix:")

print(cm)


cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.figure(
    figsize=(10, 8)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)


plt.xlabel(
    "Predicted Emotion"
)


plt.ylabel(
    "True Emotion"
)


plt.title(
    "CREMA-D Emotion Detection - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    cm_path,
    dpi=200
)


plt.close()


# ================================================================
# ACCURACY CURVE
# ================================================================

history_dict = history.history


accuracy_plot_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_dict["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history_dict["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "CREMA-D Emotion Model Accuracy"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    accuracy_plot_path,
    dpi=200
)


plt.close()


# ================================================================
# LOSS CURVE
# ================================================================

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_dict["loss"],
    label="Training Loss"
)


plt.plot(
    history_dict["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "CREMA-D Emotion Model Loss"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    loss_plot_path,
    dpi=200
)


plt.close()


# ================================================================
# PREDICTION DISTRIBUTION
# ================================================================

print("\nPrediction distribution:")


unique, counts = np.unique(
    y_pred,
    return_counts=True
)


for class_id in range(
    num_classes
):

    count = 0

    if class_id in unique:

        position = np.where(
            unique == class_id
        )[0][0]

        count = counts[position]


    percentage = (
        count / len(y_pred)
    ) * 100


    print(
        f"{class_names[class_id]:10s}: "
        f"{count:3d} "
        f"({percentage:.2f}%)"
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)


print("\nBest model:")

print(
    best_model_path
)


print("\nFinal model:")

print(
    final_model_path
)


print("\nClassification report:")

print(
    report_path
)


print("\nConfusion matrix:")

print(
    cm_path
)


print("\nAccuracy curve:")

print(
    accuracy_plot_path
)


print("\nLoss curve:")

print(
    loss_plot_path
)


print("\nTest Accuracy:")

print(
    f"{test_accuracy * 100:.2f}%"
)


print("\n")

print("=" * 70)