from pathlib import Path
import json

import joblib
import numpy as np

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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "age_final"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("FINAL AGE MODEL EVALUATION")
print("=" * 70)

print("\nLoading TEST features...")

X_test = np.load(
    FEATURE_ROOT / "test_features.npy"
)

y_test = np.load(
    FEATURE_ROOT / "test_labels.npy"
)

print(f"Test features: {X_test.shape}")
print(f"Test labels:   {y_test.shape}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    FEATURE_ROOT / "age_classes.json",
    "r",
    encoding="utf-8"
) as f:
    class_info = json.load(f)

if isinstance(class_info, dict):

    classes = [
        name
        for name, index
        in sorted(
            class_info.items(),
            key=lambda x: x[1]
        )
    ]

else:
    classes = list(class_info)


print("\nClasses:")

for i, name in enumerate(classes):
    print(f"  {i}: {name}")


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model_path = (
    MODEL_ROOT /
    "age_linear_svm.joblib"
)

scaler_path = (
    MODEL_ROOT /
    "age_svm_scaler.joblib"
)

print("\nLoading model:")
print(model_path)

model = joblib.load(
    model_path
)

print("\nLoading scaler:")
print(scaler_path)

scaler = joblib.load(
    scaler_path
)


# ============================================================
# SCALE TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("SCALING TEST DATA")
print("=" * 70)

X_test_scaled = scaler.transform(
    X_test
)

print("Scaling completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

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
    y_test,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 60+ SPECIFIC ANALYSIS
# ============================================================

senior_class_id = classes.index(
    "60_plus"
)

senior_actual = (
    y_test == senior_class_id
)

senior_predicted = (
    y_pred == senior_class_id
)

true_senior = np.sum(
    senior_actual & senior_predicted
)

actual_senior = np.sum(
    senior_actual
)

predicted_senior = np.sum(
    senior_predicted
)

if actual_senior > 0:

    senior_recall = (
        true_senior /
        actual_senior
    )

else:
    senior_recall = 0.0


if predicted_senior > 0:

    senior_precision = (
        true_senior /
        predicted_senior
    )

else:
    senior_precision = 0.0


print("\n" + "=" * 70)
print("60+ SENIOR CITIZEN ANALYSIS")
print("=" * 70)

print(
    f"\nActual 60+ samples    : "
    f"{actual_senior}"
)

print(
    f"Predicted 60+ samples : "
    f"{predicted_senior}"
)

print(
    f"Correct 60+           : "
    f"{true_senior}"
)

print(
    f"60+ Recall            : "
    f"{senior_recall:.4f}"
)

print(
    f"60+ Precision         : "
    f"{senior_precision:.4f}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    OUTPUT_ROOT /
    "final_age_confusion_matrix.npy"
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
# SAVE PREDICTIONS
# ============================================================

prediction_path = (
    OUTPUT_ROOT /
    "final_age_test_predictions.npy"
)

np.save(
    prediction_path,
    y_pred
)

print(
    "\nPredictions saved:"
)

print(prediction_path)


# ============================================================
# SAVE RESULTS JSON
# ============================================================

results = {
    "model": "LinearSVC",
    "model_file": str(model_path),
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "macro_f1": float(
        macro_f1
    ),
    "weighted_f1": float(
        weighted_f1
    ),
    "senior_60_plus": {
        "actual_samples": int(
            actual_senior
        ),
        "predicted_samples": int(
            predicted_senior
        ),
        "correct_predictions": int(
            true_senior
        ),
        "recall": float(
            senior_recall
        ),
        "precision": float(
            senior_precision
        )
    },
    "classes": classes
}


results_path = (
    OUTPUT_ROOT /
    "final_age_test_results.json"
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


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL AGE EVALUATION COMPLETE")
print("=" * 70)