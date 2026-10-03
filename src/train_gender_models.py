from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gender_features"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "gender"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "gender"
)

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("GENDER MODEL TRAINING")
print("=" * 75)

print("\nLoading feature data...")


X_train = np.load(
    FEATURE_ROOT / "train_features.npy"
)

y_train = np.load(
    FEATURE_ROOT / "train_labels.npy"
)

X_val = np.load(
    FEATURE_ROOT / "val_features.npy"
)

y_val = np.load(
    FEATURE_ROOT / "val_labels.npy"
)

X_test = np.load(
    FEATURE_ROOT / "test_features.npy"
)

y_test = np.load(
    FEATURE_ROOT / "test_labels.npy"
)


print("\nDataset shapes:")

print(
    f"Train: {X_train.shape}"
)

print(
    f"Validation: {X_val.shape}"
)

print(
    f"Test: {X_test.shape}"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_NAMES = {
    0: "female",
    1: "male"
}


print("\nLabel distribution:")

for label, name in LABEL_NAMES.items():

    print(
        f"{name:7s}: "
        f"train={np.sum(y_train == label)}, "
        f"val={np.sum(y_val == label)}, "
        f"test={np.sum(y_test == label)}"
    )


# ============================================================
# STANDARD SCALING
# ============================================================

print("\n" + "=" * 75)
print("FITTING STANDARD SCALER")
print("=" * 75)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)

joblib.dump(
    scaler,
    MODEL_ROOT / "gender_scaler.joblib"
)

print(
    "\nScaler saved:"
)

print(
    MODEL_ROOT / "gender_scaler.joblib"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_regression":
        LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE
        ),

    "random_forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "svm_rbf":
        SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        )
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "male_recall":
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

        "female_recall":
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            )
    }


# ============================================================
# TRAIN + VALIDATION
# ============================================================

validation_results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 75)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 75)

    start_time = time.time()

    # --------------------------------------------------------
    # Random Forest does not require scaling.
    # Using scaled data is still valid, but for consistency
    # we use the scaled representation for all models.
    # --------------------------------------------------------

    model.fit(
        X_train_scaled,
        y_train
    )

    elapsed = time.time() - start_time

    y_val_pred = model.predict(
        X_val_scaled
    )

    metrics = calculate_metrics(
        y_val,
        y_val_pred
    )

    metrics["model"] = model_name
    metrics["training_time_seconds"] = elapsed

    validation_results.append(
        metrics
    )

    trained_models[
        model_name
    ] = model

    print(
        f"\nTraining time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['precision_macro']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{metrics['recall_macro']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['f1_macro']:.4f}"
    )

    print(
        f"Female Recall: "
        f"{metrics['female_recall']:.4f}"
    )

    print(
        f"Male Recall: "
        f"{metrics['male_recall']:.4f}"
    )


# ============================================================
# VALIDATION COMPARISON
# ============================================================

results_df = pd.DataFrame(
    validation_results
)

results_df = results_df.sort_values(
    by="balanced_accuracy",
    ascending=False
)

print("\n" + "=" * 75)
print("VALIDATION MODEL COMPARISON")
print("=" * 75)

print(
    results_df[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "female_recall",
            "male_recall"
        ]
    ].to_string(
        index=False
    )
)


results_df.to_csv(
    OUTPUT_ROOT
    / "gender_validation_comparison.csv",
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 75)

print(
    f"SELECTED MODEL: {best_model_name}"
)

print("=" * 75)

print(
    "Selection criterion: "
    "highest validation balanced accuracy"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = (
    MODEL_ROOT
    / "gender_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)

print(
    "\nBest model saved:"
)

print(
    best_model_path
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

y_test_pred = best_model.predict(
    X_test_scaled
)

test_metrics = calculate_metrics(
    y_test,
    y_test_pred
)

print(
    f"\nModel: {best_model_name}"
)

print(
    f"Test Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Test Macro Precision: "
    f"{test_metrics['precision_macro']:.4f}"
)

print(
    f"Test Macro Recall: "
    f"{test_metrics['recall_macro']:.4f}"
)

print(
    f"Test Macro F1: "
    f"{test_metrics['f1_macro']:.4f}"
)

print(
    f"Test Female Recall: "
    f"{test_metrics['female_recall']:.4f}"
)

print(
    f"Test Male Recall: "
    f"{test_metrics['male_recall']:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    y_test_pred,
    target_names=[
        "female",
        "male"
    ],
    digits=4,
    zero_division=0
)

print(
    report
)


with open(
    OUTPUT_ROOT
    / "gender_classification_report.txt",
    "w"
) as f:

    f.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted"
)

print(
    "\n          Female   Male"
)

print(
    f"Female    {cm[0,0]:6d}   {cm[0,1]:6d}"
)

print(
    f"Male      {cm[1,0]:6d}   {cm[1,1]:6d}"
)


np.save(
    OUTPUT_ROOT
    / "gender_test_confusion_matrix.npy",
    cm
)

np.save(
    OUTPUT_ROOT
    / "gender_test_predictions.npy",
    y_test_pred
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_results = {
    "selected_model": best_model_name,
    "test_metrics": test_metrics,
    "confusion_matrix": cm.tolist(),
    "label_mapping": {
        "0": "female",
        "1": "male"
    }
}


with open(
    OUTPUT_ROOT
    / "gender_final_results.json",
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


# ============================================================
# SAVE MODEL CONFIG
# ============================================================

config = {
    "feature_dimension": int(
        X_train.shape[1]
    ),
    "label_mapping": {
        "female": 0,
        "male": 1
    },
    "sample_rate": 16000,
    "audio_duration_seconds": 5,
    "speaker_independent_split": True,
    "random_state": RANDOM_STATE
}


with open(
    MODEL_ROOT
    / "gender_config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


print("\n" + "=" * 75)
print("GENDER MODEL TRAINING COMPLETE")
print("=" * 75)

print(
    "\nFinal model:"
)

print(
    best_model_path
)

print(
    "\nFinal test results:"
)

print(
    OUTPUT_ROOT
    / "gender_final_results.json"
)