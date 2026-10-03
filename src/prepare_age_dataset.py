import csv
import io
import json
import random
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARCHIVE_PATH = Path(
    r"C:\Users\hasibul shaikh\Downloads\1784540315873-cv26-american-english-male.tar.gz"
)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "age"

METADATA_ROOT = PROJECT_ROOT / "data" / "metadata" / "common_voice_age"

RANDOM_SEED = 42

# Maximum number of training clips contributed by one speaker
MAX_CLIPS_PER_SPEAKER = 100

# Target number of training clips per age class
TARGET_TRAIN_PER_CLASS = 10000

# ============================================================
# AGE MAPPING
# ============================================================

AGE_MAPPING = {
    "teens": "00_19",
    "twenties": "20_29",
    "thirties": "30_39",
    "fourties": "40_49",
    "fifties": "50_59",
    "sixties": "60_plus",
    "seventies": "60_plus",
    "eighties": "60_plus",
    "nineties": "60_plus",
}

AGE_CLASS_NAMES = [
    "00_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_plus",
]

# ============================================================
# FOLDERS
# ============================================================

TRAIN_AUDIO_ROOT = OUTPUT_ROOT / "train"
DEV_AUDIO_ROOT = OUTPUT_ROOT / "dev"
TEST_AUDIO_ROOT = OUTPUT_ROOT / "test"

METADATA_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================


def read_tsv_from_archive(tar, filename):
    """
    Read a TSV file directly from the .tar.gz archive.
    Nothing is extracted to disk.
    """

    member = tar.getmember(filename)

    file_obj = tar.extractfile(member)

    if file_obj is None:
        raise RuntimeError(f"Could not read {filename} from archive.")

    text_stream = io.TextIOWrapper(
        file_obj,
        encoding="utf-8",
        newline=""
    )

    reader = csv.DictReader(
        text_stream,
        delimiter="\t"
    )

    rows = list(reader)

    text_stream.close()

    return rows


def convert_age(age):
    """
    Convert Common Voice age category into our six classes.
    """

    if not age:
        return None

    age = age.strip().lower()

    return AGE_MAPPING.get(age)


def prepare_rows(rows, split):
    """
    Clean metadata and add our age_class and split columns.
    """

    prepared = []

    for row in rows:

        original_age = (row.get("age") or "").strip()

        age_class = convert_age(original_age)

        # Ignore missing / unknown age
        if age_class is None:
            continue

        client_id = (row.get("client_id") or "").strip()
        path = (row.get("path") or "").strip()

        if not client_id or not path:
            continue

        prepared.append({
            "client_id": client_id,
            "path": path,
            "original_age": original_age,
            "age_class": age_class,
            "split": split,
        })

    return prepared


def group_by_class(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["age_class"]].append(row)

    return grouped


def select_balanced_training_rows(rows):
    """
    Select a balanced training subset.

    Each speaker contributes at most MAX_CLIPS_PER_SPEAKER.
    """

    print("\nPreparing balanced training dataset...")

    class_groups = group_by_class(rows)

    selected = []

    rng = random.Random(RANDOM_SEED)

    for age_class in AGE_CLASS_NAMES:

        class_rows = class_groups.get(age_class, [])

        print(
            f"\nClass {age_class}: "
            f"{len(class_rows):,} available clips"
        )

        # ----------------------------------------------------
        # Group recordings by speaker
        # ----------------------------------------------------

        speaker_groups = defaultdict(list)

        for row in class_rows:
            speaker_groups[row["client_id"]].append(row)

        # ----------------------------------------------------
        # Shuffle recordings within every speaker
        # ----------------------------------------------------

        for speaker_rows in speaker_groups.values():
            rng.shuffle(speaker_rows)

        # ----------------------------------------------------
        # Cap every speaker
        # ----------------------------------------------------

        capped_rows = []

        for speaker_id, speaker_rows in speaker_groups.items():

            usable = speaker_rows[:MAX_CLIPS_PER_SPEAKER]

            capped_rows.extend(usable)

        print(
            f"After {MAX_CLIPS_PER_SPEAKER} clips/speaker cap: "
            f"{len(capped_rows):,}"
        )

        # ----------------------------------------------------
        # Balance class
        # ----------------------------------------------------

        if len(capped_rows) < TARGET_TRAIN_PER_CLASS:

            print(
                f"WARNING: Only {len(capped_rows):,} usable "
                f"clips available for {age_class}."
            )

            selected_class = capped_rows

        else:

            rng.shuffle(capped_rows)

            selected_class = capped_rows[
                :TARGET_TRAIN_PER_CLASS
            ]

        print(
            f"Selected for {age_class}: "
            f"{len(selected_class):,}"
        )

        selected.extend(selected_class)

    rng.shuffle(selected)

    return selected


def save_metadata(rows, output_file):

    fieldnames = [
        "client_id",
        "path",
        "original_age",
        "age_class",
        "split",
    ]

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def extract_audio_files(tar, rows, output_root):

    print(
        f"\nExtracting {len(rows):,} audio files..."
    )

    extracted = 0
    missing = 0

    # Lookup required files
    required_paths = {}

    for row in rows:

        archive_path = (
            "clips/"
            + row["path"]
        )

        required_paths[archive_path] = row

    # --------------------------------------------------------
    # Iterate archive only once
    # --------------------------------------------------------

    for member in tar:

        if not member.isfile():
            continue

        if member.name not in required_paths:
            continue

        row = required_paths[member.name]

        age_class = row["age_class"]

        destination_dir = (
            output_root
            / age_class
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_file = (
            destination_dir
            / row["path"]
        )

        source = tar.extractfile(member)

        if source is None:
            missing += 1
            continue

        with open(
            destination_file,
            "wb"
        ) as out:

            shutil.copyfileobj(
                source,
                out
            )

        extracted += 1

        if extracted % 500 == 0:
            print(
                f"Extracted: "
                f"{extracted:,}/{len(rows):,}"
            )

    print(
        f"\nExtraction complete."
    )

    print(
        f"Successfully extracted: {extracted:,}"
    )

    print(
        f"Missing/unreadable: {missing:,}"
    )

    return extracted


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("COMMON VOICE AGE DATASET PREPARATION")
    print("=" * 70)

    print(
        f"\nArchive:\n{ARCHIVE_PATH}"
    )

    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )

    if not ARCHIVE_PATH.exists():

        raise FileNotFoundError(
            f"\nArchive not found:\n{ARCHIVE_PATH}"
        )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    for root in [
        TRAIN_AUDIO_ROOT,
        DEV_AUDIO_ROOT,
        TEST_AUDIO_ROOT
    ]:

        root.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Open archive
    # --------------------------------------------------------

    print("\nOpening archive...")

    with tarfile.open(
        ARCHIVE_PATH,
        mode="r:gz"
    ) as tar:

        # ====================================================
        # READ TRAIN
        # ====================================================

        print("\nReading train.tsv...")

        train_raw = read_tsv_from_archive(
            tar,
            "train.tsv"
        )

        train_rows = prepare_rows(
            train_raw,
            "train"
        )

        print(
            f"Usable training rows: "
            f"{len(train_rows):,}"
        )

        # ====================================================
        # READ DEV
        # ====================================================

        print("\nReading dev.tsv...")

        dev_raw = read_tsv_from_archive(
            tar,
            "dev.tsv"
        )

        dev_rows = prepare_rows(
            dev_raw,
            "dev"
        )

        print(
            f"Usable validation rows: "
            f"{len(dev_rows):,}"
        )

        # ====================================================
        # READ TEST
        # ====================================================

        print("\nReading test.tsv...")

        test_raw = read_tsv_from_archive(
            tar,
            "test.tsv"
        )

        test_rows = prepare_rows(
            test_raw,
            "test"
        )

        print(
            f"Usable test rows: "
            f"{len(test_rows):,}"
        )

        # ====================================================
        # BALANCED TRAINING SELECTION
        # ====================================================

        selected_train = select_balanced_training_rows(
            train_rows
        )

        # ====================================================
        # SAVE METADATA FIRST
        # ====================================================

        train_metadata = (
            METADATA_ROOT
            / "age_train.csv"
        )

        dev_metadata = (
            METADATA_ROOT
            / "age_dev.csv"
        )

        test_metadata = (
            METADATA_ROOT
            / "age_test.csv"
        )

        save_metadata(
            selected_train,
            train_metadata
        )

        save_metadata(
            dev_rows,
            dev_metadata
        )

        save_metadata(
            test_rows,
            test_metadata
        )

        print(
            f"\nSaved metadata:"
        )

        print(train_metadata)
        print(dev_metadata)
        print(test_metadata)

        # ====================================================
        # SAVE CLASS MAPPING
        # ====================================================

        class_mapping_file = (
            METADATA_ROOT
            / "age_classes.json"
        )

        with open(
            class_mapping_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                AGE_MAPPING,
                f,
                indent=4
            )

        # ====================================================
        # EXTRACT SELECTED TRAIN AUDIO
        # ====================================================

        print(
            "\n----------------------------------------"
        )

        print(
            "Extracting selected TRAIN audio..."
        )

        extract_audio_files(
            tar,
            selected_train,
            TRAIN_AUDIO_ROOT
        )

    # ========================================================
    # DEV + TEST EXTRACTION
    # ========================================================

    print(
        "\nOpening archive again for DEV/TEST..."
    )

    with tarfile.open(
        ARCHIVE_PATH,
        mode="r:gz"
    ) as tar:

        print(
            "\nExtracting DEV audio..."
        )

        extract_audio_files(
            tar,
            dev_rows,
            DEV_AUDIO_ROOT
        )

    with tarfile.open(
        ARCHIVE_PATH,
        mode="r:gz"
    ) as tar:

        print(
            "\nExtracting TEST audio..."
        )

        extract_audio_files(
            tar,
            test_rows,
            TEST_AUDIO_ROOT
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTraining clips: "
        f"{len(selected_train):,}"
    )

    print(
        f"Validation clips: "
        f"{len(dev_rows):,}"
    )

    print(
        f"Test clips: "
        f"{len(test_rows):,}"
    )

    print(
        "\nAge classes:"
    )

    for age_class in AGE_CLASS_NAMES:
        count = sum(
            1
            for row in selected_train
            if row["age_class"] == age_class
        )

        print(
            f"  {age_class}: {count:,}"
        )

    print(
        "\nDataset location:"
    )

    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()