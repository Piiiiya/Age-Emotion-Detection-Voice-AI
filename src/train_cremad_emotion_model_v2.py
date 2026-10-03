# ============================================================
# CREMA-D Emotion Model V2
# Custom CNN + SpecAugment + Label Smoothing + AdamW
# Speaker-independent split
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "emotion_cremad"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_v2"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. GPU CHECK
# ============================================================

print("=" * 70)
print("CREMA-D EMOTION MODEL V2")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("GPU not detected - using CPU")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading processed CREMA-D data...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 5. LOAD CLASS MAPPING
# ============================================================

classes_path = os.path.join(
    DATA_DIR,
    "classes.json"
)

with open(classes_path, "r") as f:
    raw_classes = json.load(f)


# Handle:
# {"angry": 0, "disgust": 1, ...}
#
# OR:
# {"0": "angry", "1": "disgust", ...}

if all(isinstance(v, int) for v in raw_classes.values()):

    class_names = [
        name
        for name, index in sorted(
            raw_classes.items(),
            key=lambda x: x[1]
        )
    ]

else:

    class_names = [
        raw_classes[str(i)]
        for i in range(len(raw_classes))
    ]


NUM_CLASSES = len(class_names)


print("\nEmotion classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# 6. DATA TYPE
# ============================================================

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

y_train = y_train.astype("int32")
y_val = y_val.astype("int32")
y_test = y_test.astype("int32")


# ============================================================
# 7. DATASET PIPELINE
# ============================================================

BATCH_SIZE = 32


train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(X_train),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# 8. SPEC AUGMENTATION
# ============================================================

@keras.utils.register_keras_serializable()
class SpecAugment(layers.Layer):

    def __init__(
        self,
        freq_mask_param=18,
        time_mask_param=28,
        num_masks=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def call(self, inputs, training=None):

        if not training:
            return inputs

        x = inputs

        shape = tf.shape(x)

        batch_size = shape[0]
        freq_bins = shape[1]
        time_bins = shape[2]

        for _ in range(self.num_masks):

            # ------------------------------
            # Frequency masking
            # ------------------------------

            freq_width = tf.random.uniform(
                [],
                minval=0,
                maxval=self.freq_mask_param + 1,
                dtype=tf.int32
            )

            freq_start = tf.random.uniform(
                [],
                minval=0,
                maxval=tf.maximum(
                    1,
                    freq_bins - freq_width + 1
                ),
                dtype=tf.int32
            )

            freq_range = tf.range(freq_bins)

            freq_mask = tf.logical_and(
                freq_range >= freq_start,
                freq_range < freq_start + freq_width
            )

            freq_mask = tf.cast(
                freq_mask,
                tf.float32
            )

            freq_mask = tf.reshape(
                freq_mask,
                [1, freq_bins, 1, 1]
            )

            x = x * (1.0 - freq_mask)


            # ------------------------------
            # Time masking
            # ------------------------------

            time_width = tf.random.uniform(
                [],
                minval=0,
                maxval=self.time_mask_param + 1,
                dtype=tf.int32
            )

            time_start = tf.random.uniform(
                [],
                minval=0,
                maxval=tf.maximum(
                    1,
                    time_bins - time_width + 1
                ),
                dtype=tf.int32
            )

            time_range = tf.range(time_bins)

            time_mask = tf.logical_and(
                time_range >= time_start,
                time_range < time_start + time_width
            )

            time_mask = tf.cast(
                time_mask,
                tf.float32
            )

            time_mask = tf.reshape(
                time_mask,
                [1, 1, time_bins, 1]
            )

            x = x * (1.0 - time_mask)

        return x

    def get_config(self):

        config = super().get_config()

        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "num_masks": self.num_masks
        })

        return config


# ============================================================
# 9. BUILD CUSTOM CNN
# ============================================================

def build_model():

    inputs = keras.Input(
        shape=X_train.shape[1:],
        name="mel_spectrogram"
    )

    # ----------------------------------------
    # SpecAugment
    # ----------------------------------------

    x = SpecAugment(
        freq_mask_param=18,
        time_mask_param=28,
        num_masks=2,
        name="spec_augment"
    )(inputs)

    # ----------------------------------------
    # Small input noise
    # ----------------------------------------

    x = layers.GaussianNoise(
        0.03
    )(x)

    # ----------------------------------------
    # Block 1
    # ----------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.25
    )(x)


    # ----------------------------------------
    # Block 2
    # ----------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.30
    )(x)


    # ----------------------------------------
    # Block 3
    # ----------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.40
    )(x)


    # ----------------------------------------
    # Global Average Pooling
    # ----------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # ----------------------------------------
    # Dense layer
    # ----------------------------------------

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(2e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.50
    )(x)


    # ----------------------------------------
    # Output
    # ----------------------------------------

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CREMA_D_Emotion_CNN_V2"
    )

    return model


model = build_model()


# ============================================================
# 10. DISPLAY MODEL
# ============================================================

print("\n")
model.summary()


# ============================================================
# 11. LOSS
# ============================================================

loss_function = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
)


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = keras.optimizers.AdamW(
    learning_rate=0.0003,
    weight_decay=1e-4
)


# ============================================================
# 13. COMPILE
# ============================================================

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# 14. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_v2_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_v2_final.keras"
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


csv_logger = keras.callbacks.CSVLogger(
    history_path
)


# ============================================================
# 15. TRAIN
# ============================================================

EPOCHS = 40

print("\n")
print("=" * 70)
print("STARTING CREMA-D EMOTION MODEL V2 TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],
    verbose=1
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)

print("\nBest model saved:")
print(best_model_path)


# ============================================================
# 17. BEST EPOCH
# ============================================================

best_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    )
) + 1

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n")
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print("Best epoch:", best_epoch)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

print("\nLoading best V2 model...")

best_model = keras.models.load_model(
    best_model_path
)


# ============================================================
# 19. TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 20. PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_probability = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "CREMA-D Emotion Model V2\n\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.figure(
    figsize=(9, 7)
)

plt.imshow(cm)

plt.title(
    "CREMA-D Emotion Model V2 - Confusion Matrix"
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "True Emotion"
)

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
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

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


# ============================================================
# 23. PREDICTION DISTRIBUTION
# ============================================================

unique, counts = np.unique(
    y_pred,
    return_counts=True
)

print("\n")
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_id, count in zip(unique, counts):

    percentage = (
        count / len(y_pred)
    ) * 100

    print(
        f"{class_names[class_id]:10s}: "
        f"{count:3d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 24. ACCURACY CURVE
# ============================================================

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
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

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "CREMA-D Emotion Model V2 Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    accuracy_path,
    dpi=200
)

plt.close()


# ============================================================
# 25. LOSS CURVE
# ============================================================

loss_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

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

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "CREMA-D Emotion Model V2 Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    loss_path,
    dpi=200
)

plt.close()


# ============================================================
# 26. SAVE CLASS MAPPING
# ============================================================

mapping_path = os.path.join(
    MODEL_DIR,
    "emotion_classes_v2.json"
)

mapping = {
    str(i): class_names[i]
    for i in range(NUM_CLASSES)
}

with open(
    mapping_path,
    "w"
) as f:

    json.dump(
        mapping,
        f,
        indent=4
    )


# ============================================================
# 27. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("CREMA-D EMOTION MODEL V2 COMPLETE")
print("=" * 70)

print("\nBest validation accuracy:")
print(
    f"{best_val_accuracy * 100:.2f}%"
)

print("\nTest accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\nFiles created:")

print(
    "Best model:",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)

print(
    "History:",
    history_path
)

print(
    "Report:",
    report_path
)

print(
    "Confusion matrix:",
    cm_path
)

print(
    "Accuracy curve:",
    accuracy_path
)

print(
    "Loss curve:",
    loss_path
)

print("\n")