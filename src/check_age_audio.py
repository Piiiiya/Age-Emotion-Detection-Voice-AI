from pathlib import Path
import random
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "age"

CLASSES = [
    "00_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_plus",
]

SAMPLES_PER_CLASS = 3
TARGET_SR = 16000

print("=" * 70)
print("AGE DATASET AUDIO QUALITY CHECK")
print("=" * 70)

total_checked = 0
total_failed = 0

for split in ["train", "dev", "test"]:

    print(f"\n{'-' * 70}")
    print(f"{split.upper()}")
    print(f"{'-' * 70}")

    for class_name in CLASSES:

        folder = DATA_ROOT / split / class_name
        files = list(folder.glob("*.mp3"))

        if not files:
            print(f"{class_name}: NO FILES")
            continue

        samples = random.sample(
            files,
            min(SAMPLES_PER_CLASS, len(files))
        )

        print(f"\n{class_name} ({len(files)} files)")

        for audio_file in samples:

            total_checked += 1

            try:
                audio, sr = librosa.load(
                    audio_file,
                    sr=TARGET_SR,
                    mono=True
                )

                duration = len(audio) / sr

                print(
                    f"  OK  | "
                    f"{audio_file.name[:30]:30s} | "
                    f"SR={sr} | "
                    f"Duration={duration:.2f}s | "
                    f"Samples={len(audio)}"
                )

            except Exception as e:

                total_failed += 1

                print(
                    f"  FAIL | "
                    f"{audio_file.name[:30]:30s} | "
                    f"{e}"
                )

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)

print(f"Files checked : {total_checked}")
print(f"Successful    : {total_checked - total_failed}")
print(f"Failed        : {total_failed}")

if total_failed == 0:
    print("\n✓ All sampled audio files loaded successfully.")
else:
    print(f"\n⚠ {total_failed} sampled files failed.")

print("=" * 70)