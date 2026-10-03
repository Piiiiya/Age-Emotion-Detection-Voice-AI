from pathlib import Path
import sys

from age_predictor import predict_age


# ============================================================
# CHANGE THIS PATH
# ============================================================

AUDIO_PATH = Path(
    r"C:\Users\hasibul shaikh\Documents\Age_Emotion_Detection_Voice_AI\data\processed\age\test\60_plus"
)


# ============================================================
# FIND FIRST AUDIO FILE
# ============================================================

audio_files = []

for extension in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
    audio_files.extend(
        AUDIO_PATH.glob(extension)
    )

if not audio_files:
    print("No audio file found.")
    print(f"Checked: {AUDIO_PATH}")
    sys.exit(1)


audio_file = audio_files[0]

print("=" * 60)
print("AGE PREDICTION TEST")
print("=" * 60)

print(f"Audio file: {audio_file}")
print()

try:

    result = predict_age(
        audio_file
    )

    print("Prediction successful!")
    print()

    print(
        f"Predicted class : "
        f"{result['class_id']}"
    )

    print(
        f"Age group       : "
        f"{result['age_group']}"
    )

    print(
        f"Senior Citizen  : "
        f"{result['is_senior']}"
    )

except Exception as e:

    print()
    print("Prediction failed.")
    print(
        f"Error: {e}"
    )

print("=" * 60)