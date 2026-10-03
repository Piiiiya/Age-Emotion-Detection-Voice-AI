import os
import json
import warnings

import numpy as np
import joblib

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
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm_v5"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm_v5"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

C = 3
GAMMA = "scale"
CLASS_WEIGHT = "balanced"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CREMA-D EMOTION MODEL V5")
print("ENHANCED 1695-FEATURE SVM")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading V5 features...")


X_train = np.load(
    os.path.join(
        FEATURE_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        FEATURE_DIR,
        "y_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        FEATURE_DIR,
        "X_val.npy"
    )
)

y_val = np.load(
    os.path.join(
        FEATURE_DIR,
        "y_val.npy"
    )
)

X_test = np.load(
    os.path.join(
        FEATURE_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        FEATURE_DIR,
        "y_test.npy"
    )
)


print("\nDataset:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if X_train.ndim != 2:
    raise ValueError(
        f"X_train must be 2D, got {X_train.shape}"
    )

if X_val.ndim != 2:
    raise ValueError(
        f"X_val must be 2D, got {X_val.shape}"
    )

if X_test.ndim != 2:
    raise ValueError(
        f"X_test must be 2D, got {X_test.shape}"
    )


if len(X_train) != len(y_train):
    raise ValueError(
        "Training feature/label mismatch."
    )

if len(X_val) != len(y_val):
    raise ValueError(
        "Validation feature/label mismatch."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Test feature/label mismatch."
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\n")
print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print(
    "\nSVM kernel: RBF"
)

print(
    "C:",
    C
)

print(
    "Gamma:",
    GAMMA
)

print(
    "Class weight:",
    CLASS_WEIGHT
)


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
                C=C,
                gamma=GAMMA,
                class_weight=CLASS_WEIGHT,
                probability=True,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING V5 SVM")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Features per sample:",
    X_train.shape[1]
)

print(
    "\nPlease wait..."
)


model.fit(
    X_train,
    y_train
)


print(
    "\nTraining completed."
)


# ============================================================
# VALIDATION
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
# TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


print(
    "\nThe test set is evaluated only now."
)


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
# CLASSIFICATION REPORT
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
# CONFUSION MATRIX
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
# PREDICTION DISTRIBUTION
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
        count / len(test_pred)
    ) * 100

    print(
        f"{CLASS_NAMES[class_id]:10s} "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_svm_v5.joblib"
)


joblib.dump(
    model,
    model_path
)


print("\n")
print(
    "Model saved:"
)

print(
    model_path
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

classes_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_svm_v5_classes.json"
)


class_mapping = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


with open(
    classes_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=4
    )


# ============================================================
# SAVE REPORT
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
        "CREMA-D EMOTION MODEL V5\n"
    )

    f.write(
        "Enhanced 1695-Feature RBF SVM\n\n"
    )

    f.write(
        f"C = {C}\n"
    )

    f.write(
        f"gamma = {GAMMA}\n"
    )

    f.write(
        f"class_weight = {CLASS_WEIGHT}\n\n"
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
# SAVE CONFUSION MATRIX
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
# SAVE SUMMARY
# ============================================================

summary = {
    "model": "RBF SVM V5",
    "feature_count": int(
        X_train.shape[1]
    ),
    "train_samples": int(
        len(X_train)
    ),
    "validation_samples": int(
        len(X_val)
    ),
    "test_samples": int(
        len(X_test)
    ),
    "C": C,
    "gamma": GAMMA,
    "class_weight": CLASS_WEIGHT,
    "validation_accuracy":
        float(val_accuracy),
    "validation_balanced_accuracy":
        float(val_balanced_accuracy),
    "test_accuracy":
        float(test_accuracy),
    "test_balanced_accuracy":
        float(test_balanced_accuracy)
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)

print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)

print(
    "Validation Balanced Accuracy:",
    f"{val_balanced_accuracy * 100:.2f}%"
)

print(
    "\nTEST Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print(
    "TEST Balanced Accuracy:",
    f"{test_balanced_accuracy * 100:.2f}%"
)

print(
    "\nModel:"
)

print(
    model_path
)

print(
    "\n"
    + "=" * 70
)