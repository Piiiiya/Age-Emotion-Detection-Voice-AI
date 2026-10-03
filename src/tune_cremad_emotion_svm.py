# ============================================================
# CREMA-D EMOTION MODEL V4
# SVM HYPERPARAMETER TUNING
#
# IMPORTANT:
# Validation set -> used for model selection
# Test set       -> untouched until final evaluation
# ============================================================

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
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "emotion_cremad_svm_tuning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

CLASS_NAMES = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad"
]


# ============================================================
# 3. LOAD FEATURES
# ============================================================

print("=" * 70)
print("CREMA-D EMOTION MODEL V4")
print("SVM HYPERPARAMETER TUNING")
print("=" * 70)


print("\nLoading previously extracted features...")


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
# 4. SAFETY CHECK
# ============================================================

if X_train.ndim != 2:
    raise ValueError(
        f"X_train must be 2D. Got {X_train.shape}"
    )

if X_val.ndim != 2:
    raise ValueError(
        f"X_val must be 2D. Got {X_val.shape}"
    )

if X_test.ndim != 2:
    raise ValueError(
        f"X_test must be 2D. Got {X_test.shape}"
    )


if len(X_train) != len(y_train):
    raise ValueError(
        "Training feature/label count mismatch."
    )

if len(X_val) != len(y_val):
    raise ValueError(
        "Validation feature/label count mismatch."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Test feature/label count mismatch."
    )


# ============================================================
# 5. HYPERPARAMETER SEARCH
# ============================================================

# We deliberately keep this search reasonably small
# because RBF SVM can be computationally expensive.

C_VALUES = [
    0.3,
    1,
    3,
    10,
    30,
    100
]


GAMMA_VALUES = [
    "scale",
    0.0001,
    0.0003,
    0.001,
    0.003
]


CLASS_WEIGHT_VALUES = [
    None,
    "balanced"
]


# ============================================================
# 6. RESULTS STORAGE
# ============================================================

results = []

best_model = None

best_params = None

best_val_accuracy = -1

best_val_balanced_accuracy = -1


# ============================================================
# 7. TRAIN MODELS
# ============================================================

total_experiments = (
    len(C_VALUES)
    *
    len(GAMMA_VALUES)
    *
    len(CLASS_WEIGHT_VALUES)
)


experiment_number = 0


print("\n")
print("=" * 70)
print("STARTING VALIDATION-BASED HYPERPARAMETER SEARCH")
print("=" * 70)

print(
    f"\nTotal configurations: {total_experiments}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The test set will NOT be used during tuning."
)


for C in C_VALUES:

    for gamma in GAMMA_VALUES:

        for class_weight in CLASS_WEIGHT_VALUES:

            experiment_number += 1


            print("\n")
            print("-" * 70)

            print(
                f"Experiment "
                f"{experiment_number}/{total_experiments}"
            )

            print(
                f"C = {C}"
            )

            print(
                f"gamma = {gamma}"
            )

            print(
                f"class_weight = {class_weight}"
            )


            # ------------------------------------------------
            # Build pipeline
            # ------------------------------------------------

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
                            gamma=gamma,
                            probability=True,
                            class_weight=class_weight,
                            random_state=SEED
                        )
                    )
                ]
            )


            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            print(
                "\nTraining..."
            )

            model.fit(
                X_train,
                y_train
            )


            # ------------------------------------------------
            # Validation prediction
            # ------------------------------------------------

            val_pred = model.predict(
                X_val
            )


            # ------------------------------------------------
            # Validation metrics
            # ------------------------------------------------

            val_accuracy = accuracy_score(
                y_val,
                val_pred
            )


            val_balanced_accuracy = balanced_accuracy_score(
                y_val,
                val_pred
            )


            print(
                "Validation Accuracy:",
                f"{val_accuracy * 100:.2f}%"
            )

            print(
                "Validation Balanced Accuracy:",
                f"{val_balanced_accuracy * 100:.2f}%"
            )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                {
                    "C": C,
                    "gamma": str(gamma),
                    "class_weight": str(class_weight),
                    "validation_accuracy": val_accuracy,
                    "validation_balanced_accuracy":
                        val_balanced_accuracy
                }
            )


            # ------------------------------------------------
            # Select best model
            #
            # Primary:
            # validation balanced accuracy
            #
            # Secondary:
            # validation accuracy
            # ------------------------------------------------

            is_better = False


            if (
                val_balanced_accuracy
                >
                best_val_balanced_accuracy
            ):

                is_better = True


            elif (
                np.isclose(
                    val_balanced_accuracy,
                    best_val_balanced_accuracy
                )
                and
                val_accuracy
                >
                best_val_accuracy
            ):

                is_better = True


            if is_better:

                best_val_balanced_accuracy = (
                    val_balanced_accuracy
                )

                best_val_accuracy = (
                    val_accuracy
                )

                best_model = model

                best_params = {
                    "C": C,
                    "gamma": gamma,
                    "class_weight": class_weight
                }


                print(
                    "\n*** NEW BEST MODEL ***"
                )

                print(
                    "Best validation accuracy:",
                    f"{best_val_accuracy * 100:.2f}%"
                )

                print(
                    "Best validation balanced accuracy:",
                    f"{best_val_balanced_accuracy * 100:.2f}%"
                )


# ============================================================
# 8. SAVE TUNING RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HYPERPARAMETER SEARCH COMPLETE")
print("=" * 70)


results_sorted = sorted(
    results,
    key=lambda x: (
        x["validation_balanced_accuracy"],
        x["validation_accuracy"]
    ),
    reverse=True
)


results_csv = os.path.join(
    OUTPUT_DIR,
    "svm_tuning_results.csv"
)


import csv


with open(
    results_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "C",
            "gamma",
            "class_weight",
            "validation_accuracy",
            "validation_balanced_accuracy"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results_sorted
    )


# ============================================================
# 9. DISPLAY TOP RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 10 VALIDATION CONFIGURATIONS")
print("=" * 70)


for rank, result in enumerate(
    results_sorted[:10],
    start=1
):

    print(
        f"\n#{rank}"
    )

    print(
        f"C={result['C']}, "
        f"gamma={result['gamma']}, "
        f"class_weight={result['class_weight']}"
    )

    print(
        "Validation Accuracy:",
        f"{result['validation_accuracy'] * 100:.2f}%"
    )

    print(
        "Validation Balanced Accuracy:",
        f"{result['validation_balanced_accuracy'] * 100:.2f}%"
    )


# ============================================================
# 10. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)


print(
    "\nBest parameters:"
)

print(
    best_params
)


print(
    "\nBest validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    "Best validation balanced accuracy:",
    f"{best_val_balanced_accuracy * 100:.2f}%"
)


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


print(
    "\nThe test set is being used NOW "
    "for the first time during this experiment."
)


test_pred = best_model.predict(
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
# 12. CLASSIFICATION REPORT
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
# 13. CONFUSION MATRIX
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
# 14. SAVE BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "emotion_cremad_svm_v4_best.joblib"
)


joblib.dump(
    best_model,
    best_model_path
)


print(
    "\nBest tuned model saved:"
)

print(
    best_model_path
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
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
        "CREMA-D EMOTION MODEL V4\n"
    )

    f.write(
        "SVM HYPERPARAMETER TUNING\n\n"
    )

    f.write(
        "Selected Parameters:\n"
    )

    f.write(
        f"{best_params}\n\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Validation Balanced Accuracy: "
        f"{best_val_balanced_accuracy * 100:.2f}%\n\n"
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
# 16. SAVE CONFUSION MATRIX
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
# 17. SAVE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.json"
)


summary = {
    "model": "RBF SVM",
    "feature_count": int(X_train.shape[1]),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "best_parameters": {
        "C": best_params["C"],
        "gamma": str(
            best_params["gamma"]
        ),
        "class_weight": str(
            best_params["class_weight"]
        )
    },
    "validation_accuracy":
        float(best_val_accuracy),
    "validation_balanced_accuracy":
        float(best_val_balanced_accuracy),
    "test_accuracy":
        float(test_accuracy),
    "test_balanced_accuracy":
        float(test_balanced_accuracy)
}


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
# 18. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("CREMA-D EMOTION MODEL V4 COMPLETE")
print("=" * 70)


print(
    "\nSelected parameters:"
)

print(
    best_params
)


print(
    "\nValidation Accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    "Validation Balanced Accuracy:",
    f"{best_val_balanced_accuracy * 100:.2f}%"
)


print(
    "\nFINAL TEST Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


print(
    "FINAL TEST Balanced Accuracy:",
    f"{test_balanced_accuracy * 100:.2f}%"
)


print(
    "\nBest model:"
)

print(
    best_model_path
)


print(
    "\nTuning results:"
)

print(
    results_csv
)


print(
    "\nClassification report:"
)

print(
    report_path
)


print(
    "\n"
    + "=" * 70
)

print(
    "V4 FINISHED SUCCESSFULLY"
)

print(
    "=" * 70
)