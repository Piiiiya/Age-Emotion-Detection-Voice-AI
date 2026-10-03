from pathlib import Path
import pandas as pd
import re

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CREMAD_ROOT = PROJECT_ROOT / "crema_d"
AUDIO_DIR = CREMAD_ROOT / "AudioWAV"
DEMOGRAPHICS_FILE = CREMAD_ROOT / "VideoDemographics.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_FILE = OUTPUT_DIR / "crema_d_male_metadata.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


# ============================================================
# LOAD DEMOGRAPHICS
# ============================================================

print("=" * 70)
print("CREMA-D MALE METADATA PREPARATION")
print("=" * 70)

print("\nLoading demographics...")

demographics = pd.read_csv(DEMOGRAPHICS_FILE)

print(f"Total actors in demographics: {len(demographics)}")


# Normalize column names

demographics.columns = [
    column.strip()
    for column in demographics.columns
]


# Normalize sex

demographics["Sex"] = (
    demographics["Sex"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Keep males only

male_demographics = demographics[
    demographics["Sex"] == "male"
].copy()


print(f"Male actors: {len(male_demographics)}")


# Create actor lookup

actor_lookup = {}

for _, row in male_demographics.iterrows():

    actor_id = int(row["ActorID"])

    actor_lookup[actor_id] = {
        "age": int(row["Age"]),
        "gender": "male",
        "race": row["Race"],
        "ethnicity": row["Ethnicity"],
    }


# ============================================================
# FIND AUDIO FILES
# ============================================================

print("\nScanning AudioWAV directory...")

wav_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

print(f"Total WAV files found: {len(wav_files)}")


# ============================================================
# PARSE FILENAMES
# ============================================================

records = []

failed = []

for wav_path in wav_files:

    filename = wav_path.name

    # Expected pattern:
    #
    # 1001_DFA_ANG_XX.wav
    # 1001_IEO_ANG_HI.wav
    #
    match = re.match(
        r"^(\d{4})_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav$",
        filename,
        re.IGNORECASE,
    )

    if not match:
        failed.append(
            (filename, "filename_pattern")
        )
        continue

    actor_id = int(match.group(1))

    emotion_code = match.group(3).upper()

    # Check actor exists and is male

    if actor_id not in actor_lookup:

        # Female actors are intentionally skipped

        continue

    # Check emotion

    if emotion_code not in EMOTION_MAP:

        failed.append(
            (
                filename,
                f"unknown_emotion_{emotion_code}",
            )
        )

        continue

    emotion = EMOTION_MAP[emotion_code]

    actor_info = actor_lookup[actor_id]

    records.append(
        {
            "file": str(
                wav_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),

            "filename": filename,

            "actor_id": actor_id,

            "age": actor_info["age"],

            "gender": actor_info["gender"],

            "race": actor_info["race"],

            "ethnicity": actor_info["ethnicity"],

            "emotion_code": emotion_code,

            "emotion": emotion,
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# Sort

df = df.sort_values(
    ["actor_id", "emotion", "filename"]
).reset_index(drop=True)


# Save

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nMale recordings: {len(df)}")

print(
    f"Unique male actors: {df['actor_id'].nunique()}"
)

print(
    f"Output file:\n{OUTPUT_FILE}"
)


print("\nEmotion distribution:")

print(
    df["emotion"]
    .value_counts()
    .sort_index()
)


print("\nAge distribution:")

print(
    df["age"]
    .describe()
)


print("\nMale actors:")

print(
    df[
        ["actor_id", "age"]
    ]
    .drop_duplicates()
    .sort_values("actor_id")
    .to_string(index=False)
)


print("\nSenior male actors (>60):")

senior = (
    df[
        df["age"] > 60
    ][
        ["actor_id", "age"]
    ]
    .drop_duplicates()
    .sort_values("age")
)

print(
    senior.to_string(index=False)
)


print("\nFirst 10 records:")

print(
    df.head(10).to_string(index=False)
)


print("\nFailed filename parsing:")

print(len(failed))


if failed:

    print("\nFirst 10 failures:")

    for item in failed[:10]:

        print(item)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)