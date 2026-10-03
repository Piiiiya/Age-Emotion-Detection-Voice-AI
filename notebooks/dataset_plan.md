# Age and Emotion Detection Through Voice — Dataset Plan

## 1. Gender Detection

Task:
Classify an input voice as:

- Male
- Female

Candidate dataset:
Mozilla Common Voice / VoxCeleb

Important:
The dataset must contain reliable speaker gender metadata.

---

## 2. Age Detection

Task:
Estimate the speaker's age.

Required for the assignment:

- Age <= 60 → show age only
- Age > 60 → Senior Citizen + emotion

Candidate dataset:
VoxCeleb / age-annotated VoxCeleb data

Important:
We must verify whether the dataset provides reliable age labels before training.

---

## 3. Emotion Detection

Task:
Detect emotion from a male voice.

Initial dataset:
RAVDESS Speech

Possible emotion classes:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

---

## 4. Audio Standardization

All audio should eventually be converted to:

Sample rate:
16000 Hz

Mono:
Yes

Feature representation:
Log-Mel Spectrogram / MFCC

---

## 5. Data Split

We should avoid speaker leakage.

Preferred:

Training speakers
Validation speakers
Testing speakers

should be different whenever the dataset allows it.

---

## 6. Models

Gender:
CNN on audio features

Age:
Age classification/regression model

Emotion:
CNN on audio features

---

## 7. Final Pipeline

Voice Note
    ↓
Gender Detection
    ↓
Female → "Upload male voice."
    ↓
Male
    ↓
Age Detection
    ↓
Age <= 60 → Show age
    ↓
Age > 60
    ↓
Senior Citizen
    ↓
Emotion Detection