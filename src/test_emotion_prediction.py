from pathlib import Path
import sys

# Allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from emotion_predictor import predict_emotion


# ============================================================
# FIND A REAL CREMA-D TEST AUDIO
# ============================================================

TEST_DIR = PROJECT_ROOT / "crema_d" / "AudioWAV"

audio_files = list(TEST_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        f"No CREMA-D WAV files found in:\n{TEST_DIR}"
    )


# Use the first available file
audio_path = audio_files[0]


# ============================================================
# PREDICT
# ============================================================

print("=" * 60)
print("CREMA-D EMOTION PREDICTION TEST")
print("=" * 60)

print(f"Audio file:\n{audio_path}")

result = predict_emotion(audio_path)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\nPrediction successful!")

print(f"\nClass ID          : {result['class_id']}")
print(f"Emotion           : {result['emotion']}")
print(f"Feature dimension : {result['feature_dimension']}")

if "decision_scores" in result:

    print("\nSVM decision scores:")

    for class_id, score in enumerate(result["decision_scores"]):
        emotion = {
            0: "angry",
            1: "disgust",
            2: "fearful",
            3: "happy",
            4: "neutral",
            5: "sad"
        }[class_id]

        print(
            f"  {emotion:<10} : {score:.4f}"
        )

print("\n" + "=" * 60)