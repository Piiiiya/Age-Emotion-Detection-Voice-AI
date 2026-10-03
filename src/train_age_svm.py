from pathlib import Path
import json
import time

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
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

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGE MODEL — LINEAR SVM")
print("=" * 70)

print("\nLoading feature matrices...")

X_train = np.load(FEATURE_ROOT / "train_features.npy")
y_train = np.load(FEATURE_ROOT / "train_labels.npy")

X_dev = np.load(FEATURE_ROOT / "dev_features.npy")
y_dev = np.load(FEATURE_ROOT / "dev_labels.npy")

print(f"Train: {X_train.shape}")
print(f"Dev:   {X_dev.shape}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(FEATURE_ROOT / "age_classes.json", "r", encoding="utf-8") as f:
    class_info = json.load(f)

if isinstance(class_info, dict):
    # Expected format:
    # {"00_19": 0, "20_29": 1, ...}
    if all(isinstance(v, int) for v in class_info.values()):
        classes = [
            name for name, index
            in sorted(class_info.items(), key=lambda x: x[1])
        ]
    else:
        classes = list(class_info.keys())
else:
    classes = list(class_info)

print("\nClasses:")
for i, name in enumerate(classes):
    print(f"  {i}: {name}")


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING FEATURES")
print("=" * 70)

start = time.time()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)

print(f"Scaling completed in {time.time() - start:.2f} seconds")

scaler_path = MODEL_ROOT / "age_svm_scaler.joblib"
joblib.dump(scaler, scaler_path)

print(f"Scaler saved:")
print(scaler_path)


# ============================================================
# LINEAR SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LINEAR SVM")
print("=" * 70)

start = time.time()

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=RANDOM_STATE
)

model.fit(X_train_scaled, y_train)

training_time = time.time() - start

print(f"Training time: {training_time:.2f} seconds")


# ============================================================
# DEV EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("LINEAR SVM — DEV RESULTS")
print("=" * 70)

y_pred = model.predict(X_dev_scaled)

accuracy = accuracy_score(y_dev, y_pred)
balanced_accuracy = balanced_accuracy_score(y_dev, y_pred)
macro_f1 = f1_score(y_dev, y_pred, average="macro")
weighted_f1 = f1_score(y_dev, y_pred, average="weighted")

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")

print("\nClassification Report:")

report = classification_report(
    y_dev,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_dev, y_pred)

print("Confusion Matrix:")
print(cm)

cm_path = OUTPUT_ROOT / "age_linear_svm_dev_confusion_matrix.npy"
np.save(cm_path, cm)

print(f"\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_ROOT / "age_linear_svm.joblib"

joblib.dump(model, model_path)

print("\nModel saved:")
print(model_path)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "LinearSVC",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "training_time_seconds": float(training_time),
    "classes": classes
}

results_path = OUTPUT_ROOT / "age_linear_svm_dev_results.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResults saved:")
print(results_path)

print("\n" + "=" * 70)
print("LINEAR SVM TRAINING COMPLETE")
print("=" * 70)