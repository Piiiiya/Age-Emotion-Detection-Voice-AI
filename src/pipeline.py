from pathlib import Path

from gender_predictor import predict_gender
from age_predictor import predict_age
from emotion_predictor import predict_emotion


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_voice(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    # ========================================================
    # STEP 1 — GENDER
    # ========================================================

    gender_result = predict_gender(audio_path)

    gender = gender_result["gender"]
    gender_confidence = gender_result.get("confidence")

    # --------------------------------------------------------
    # FEMALE → STOP
    # --------------------------------------------------------

    if gender == "female":

        return {
            "status": "rejected",
            "message": "Upload male voice.",
            "gender": "female",
            "gender_confidence": gender_confidence,
            "age": None,
            "age_group": None,
            "senior_citizen": False,
            "emotion": None
        }

    # ========================================================
    # STEP 2 — AGE
    # ========================================================

    age_result = predict_age(audio_path)

    age_group = age_result["age_group"]
    is_senior = age_result["is_senior"]

    # ========================================================
    # MALE + SENIOR CITIZEN
    # ========================================================

    if is_senior:

        emotion_result = predict_emotion(audio_path)

        return {
            "status": "success",
            "message": "Analysis completed.",
            "gender": "male",
            "gender_confidence": gender_confidence,
            "age": None,
            "age_group": age_group,
            "senior_citizen": True,
            "emotion": emotion_result["emotion"],
            "emotion_class_id": emotion_result["class_id"],
            "emotion_feature_dimension": emotion_result[
                "feature_dimension"
            ]
        }

    # ========================================================
    # MALE + NON-SENIOR
    # ========================================================

    return {
        "status": "success",
        "message": "Analysis completed.",
        "gender": "male",
        "gender_confidence": gender_confidence,
        "age": None,
        "age_group": age_group,
        "senior_citizen": False,
        "emotion": None
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 60)
    print("VOICE ANALYSIS RESULT")
    print("=" * 60)

    print(f"Status : {result['status']}")

    print(f"Gender : {result['gender']}")

    if result.get("gender_confidence") is not None:
        print(
            f"Gender confidence : "
            f"{result['gender_confidence']:.2f}"
        )

    # Female
    if result["status"] == "rejected":

        print("\n" + result["message"])

        print("=" * 60)

        return

    # Male
    print(f"Age group : {result['age_group']}")

    if result["senior_citizen"]:

        print("Category : Senior Citizen")

        print(
            f"Emotion : {result['emotion']}"
        )

    else:

        print("Category : Non-Senior Citizen")
        print("Emotion : Not required")

    print("=" * 60)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python src\\pipeline.py "
            "\"path_to_audio_file\""
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    result = analyze_voice(audio_file)

    print_result(result)