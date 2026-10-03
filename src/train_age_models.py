from pathlib import Path
import json
import time

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "age_features"
)

MODEL_ROOT = PROJECT_ROOT / "models" / "age"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "age_models"

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "00_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_plus",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AGE MODEL TRAINING")
print("=" * 70)

print("\nLoading feature matrices...")

X_train = np.load(
    FEATURE_ROOT / "train_features.npy",
    mmap_mode="r"
)

y_train = np.load(
    FEATURE_ROOT / "train_labels.npy"
)

X_dev = np.load(
    FEATURE_ROOT / "dev_features.npy",
    mmap_mode="r"
)

y_dev = np.load(
    FEATURE_ROOT / "dev_labels.npy"
)

X_test = np.load(
    FEATURE_ROOT / "test_features.npy",
    mmap_mode="r"
)

y_test = np.load(
    FEATURE_ROOT / "test_labels.npy"
)

print(f"Train: {X_train.shape}")
print(f"Dev:   {X_dev.shape}")
print(f"Test:  {X_test.shape}")


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING FEATURES")
print("=" * 70)

scaler = StandardScaler()

start = time.time()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_dev_scaled = scaler.transform(
    X_dev
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    f"Scaling completed in "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# SAVE SCALER
# ============================================================

scaler_path = MODEL_ROOT / "age_scaler.joblib"

joblib.dump(
    scaler,
    scaler_path
)

print(
    f"Scaler saved:\n{scaler_path}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name,
    X,
    y,
    split_name
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    print(
        f"\n{model_name} - {split_name}"
    )

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy  : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : {weighted_f1:.4f}"
    )

    if split_name == "DEV":

        print("\nClassification Report:")

        print(
            classification_report(
                y,
                predictions,
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0
            )
        )

        print("Confusion Matrix:")

        cm = confusion_matrix(
            y,
            predictions
        )

        print(cm)

        np.save(
            OUTPUT_ROOT
            / f"{model_name.lower()}_dev_confusion_matrix.npy",
            cm
        )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# RESULTS
# ============================================================

results = {}


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 — LOGISTIC REGRESSION")
print("=" * 70)

start = time.time()

logistic_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - start:.2f} seconds"
)

logistic_results = evaluate_model(
    logistic_model,
    "logistic_regression",
    X_dev_scaled,
    y_dev,
    "DEV"
)

results["logistic_regression"] = logistic_results

joblib.dump(
    logistic_model,
    MODEL_ROOT / "age_logistic_regression.joblib"
)


# ============================================================
# MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 — RANDOM FOREST")
print("=" * 70)

start = time.time()

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - start:.2f} seconds"
)

rf_results = evaluate_model(
    random_forest,
    "random_forest",
    X_dev,
    y_dev,
    "DEV"
)

results["random_forest"] = rf_results

joblib.dump(
    random_forest,
    MODEL_ROOT / "age_random_forest.joblib"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_ROOT
    / "model_comparison_dev.json"
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


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEV MODEL COMPARISON")
print("=" * 70)

for model_name, metrics in results.items():

    print(
        f"\n{model_name}"
    )

    print(
        f"  Accuracy         : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro F1         : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted F1      : "
        f"{metrics['weighted_f1']:.4f}"
    )

print("\nModels saved to:")
print(MODEL_ROOT)

print("\nResults saved to:")
print(results_path)

print("=" * 70)