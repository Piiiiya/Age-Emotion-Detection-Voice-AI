# Age and Emotion Detection Through Voice

An AI/ML-based voice analysis application that processes audio recordings to classify gender, estimate age groups, identify senior-citizen status, and detect emotion when applicable.

The application uses a machine-learning pipeline and a Streamlit graphical user interface to provide predictions from uploaded audio files.

## Project Overview

The system follows a conditional prediction workflow:

1. Upload an audio recording.
2. Predict the speaker's gender.
3. If the speaker is female, display **"Upload male voice."**
4. If the speaker is male, estimate the age group.
5. If the predicted age group is 60+, classify the speaker as a senior citizen and perform emotion detection.
6. Otherwise, display the age group without running emotion detection.

## Key Features

- Audio upload and playback through a Streamlit interface.
- Male/female voice classification.
- Age-group prediction.
- Senior-citizen classification.
- Conditional emotion detection.
- Modular prediction pipeline.
- Machine-learning model evaluation.
- Speaker-independent dataset splitting for model evaluation.

## Technology Stack

- Python
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- TensorFlow/Keras (used during model experimentation)

## Model Architecture

The application uses three prediction components:

| Component | Algorithm | Purpose |
|---|---|---|
| Gender classification | Logistic Regression | Classifies male and female voices |
| Age-group classification | Linear Support Vector Machine | Predicts one of six age groups |
| Emotion classification | RBF Support Vector Machine | Classifies vocal emotion |

### Age Groups

- 00–19
- 20–29
- 30–39
- 40–49
- 50–59
- 60+

**Age classification limitation:** The model predicts age groups, not exact ages. The available dataset uses a 60+ category, so the implementation treats this category as senior-citizen status. It cannot distinguish age 60 from ages above 60.

## Model Evaluation

The following results were obtained during evaluation on held-out test data.

| Model | Test Accuracy |
|---|---:|
| Gender classification | 98.75% |
| Age-group classification | 22.82% |
| Emotion classification | 52.26% |

These results represent the evaluated datasets and should not be interpreted as guaranteed performance on real-world recordings.

The age model has limited predictive performance, and the emotion model shows moderate performance. Further dataset improvements, model tuning, and testing on diverse real-world audio are recommended.

## Datasets

The project uses the following datasets during development and evaluation:

- **RAVDESS:** Used for gender and emotion-related audio experiments.
- **CREMA-D:** Used for emotion classification.
- **Common Voice:** Used for age-group classification.
- **VoxCeleb/AgeVoxCeleb metadata:** Used for age dataset exploration and preparation.

Dataset files are not included in the public repository. Please refer to the original dataset providers for access conditions and licensing.

## Project Structure

```text
Age_Emotion_Detection_Voice_AI/
│
├── app.py
├── src/
│   ├── audio_features.py
│   ├── gender_predictor.py
│   ├── age_predictor.py
│   ├── emotion_predictor.py
│   └── pipeline.py
│
├── models/
│   ├── age/
│   ├── gender/
│   └── emotion_cremad_svm_v5.joblib
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── outputs/
│   ├── age_final/
│   ├── gender/
│   └── emotion_cremad_svm_v5/
│
├── notebooks/
├── docs/
├── screenshots/
├── submission/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Age_Emotion_Detection_Voice_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure model files

Place the required trained model files in their expected locations. Large models and datasets may be distributed separately from the source repository.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Screenshots

Screenshots of the application and prediction results will be added to the `screenshots/` directory.

## Limitations

- Age prediction is performed by age group rather than exact age.
- The 60+ category is used as the senior-citizen threshold.
- Model performance may vary with recording quality, background noise, accents, and speaker characteristics.
- Gender classification is a binary model trained on the selected dataset and does not represent all gender identities.
- Emotion predictions are estimates based on vocal characteristics and should not be treated as definitive assessments of a person's emotional state.
- The system is intended for educational and demonstration purposes.

## Future Improvements

- Improve age-group classification through additional representative datasets.
- Improve emotion classification and evaluate on independent recordings.
- Add confidence calibration and uncertainty reporting.
- Expand audio preprocessing and noise handling.
- Evaluate performance across different accents and recording conditions.
- Improve the user interface and deployment workflow.

## Author

**Hasibul Shaikh**

MCA Graduate | AI, Machine Learning and Data Science

## Project Purpose

Developed as part of an AI/ML internship project to demonstrate dataset preparation, feature engineering, machine-learning model development, evaluation, and GUI integration.
