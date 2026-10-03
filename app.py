
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from pipeline import analyze_voice


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Age & Emotion Detection",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin: 10px 0;
    }

    .result-card {
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .result-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .result-row {
        font-size: 18px;
        margin: 10px 0;
    }

    .warning-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎙️ About the Project")

    st.write(
        """
        This application uses machine learning to analyze
        a speaker's voice.

        **Pipeline**

        1. Gender Detection
        2. Age Group Detection
        3. Senior Citizen Detection
        4. Emotion Detection
        """
    )

    st.divider()

    st.subheader("Supported Audio")

    st.write(
        """
        - WAV
        - MP3
        - OGG
        - FLAC
        - M4A
        """
    )

    st.divider()

    st.subheader("Decision Logic")

    st.write(
        """
        Female voice → Upload male voice.

        Male voice → Detect age.

        60+ → Senior Citizen + Emotion.

        Below 60 → Age only.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎙️ Age & Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Voice-Based Machine Learning System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown("### 📋 Voice Analysis")

st.write(
    """
    Upload a voice note below. The system will first determine
    whether the speaker is male or female.

    Only male voices continue to the age-analysis stage.
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "🎵 Upload Voice Note",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ],
    help="Upload a clear voice recording."
)


# ============================================================
# AUDIO PROCESSING
# ============================================================

if uploaded_file is not None:

    st.success(
        f"File uploaded: {uploaded_file.name}"
    )

    st.audio(
        uploaded_file,
        format=uploaded_file.type
    )

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Voice",
        use_container_width=True,
        type="primary"
    )

    if analyze_button:

        # ----------------------------------------------------
        # SAVE TEMPORARY FILE
        # ----------------------------------------------------

        temp_dir = PROJECT_ROOT / "outputs" / "temp"

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = None

        try:

            # Use a unique temporary filename to avoid
            # overwriting files during simultaneous requests.

            suffix = Path(uploaded_file.name).suffix.lower()

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                prefix="voice_",
                dir=temp_dir,
                delete=False
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = Path(temp_file.name)

            # ------------------------------------------------
            # RUN PIPELINE
            # ------------------------------------------------

            with st.spinner(
                "🔄 Analyzing voice... Please wait."
            ):

                result = analyze_voice(
                    temp_path
                )

            # =================================================
            # FEMALE VOICE REJECTION
            # =================================================

            if result["status"] == "rejected":

                st.error(
                    "Upload male voice."
                )

                # Do not display an Analysis Result card
                # for rejected female voices.

            # =================================================
            # MALE VOICE
            # =================================================

            else:

                gender = result["gender"]
                age_group = result["age_group"]

                st.success(
                    "✅ Voice analysis completed successfully."
                )

                # --------------------------------------------
                # SENIOR CITIZEN
                # --------------------------------------------

                if result["senior_citizen"]:

                    emotion = result["emotion"]

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="result-title">
                        🎯 Analysis Result
                        </div>

                        <div class="result-row">
                        <b>Gender:</b>
                        {gender.title()}
                        </div>

                        <div class="result-row">
                        <b>Age Group:</b>
                        {age_group}
                        </div>

                        <div class="result-row">
                        <b>Category:</b>
                        Senior Citizen
                        </div>

                        <div class="result-row">
                        <b>Emotion:</b>
                        {emotion.title()}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --------------------------------------------
                # NON-SENIOR CITIZEN
                # --------------------------------------------

                else:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="result-title">
                        🎯 Analysis Result
                        </div>

                        <div class="result-row">
                        <b>Gender:</b>
                        {gender.title()}
                        </div>

                        <div class="result-row">
                        <b>Age Group:</b>
                        {age_group}
                        </div>

                        <div class="result-row">
                        <b>Category:</b>
                        Non-Senior Citizen
                        </div>

                        <div class="result-row">
                        <b>Emotion:</b>
                        Not Required
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as error:

            st.error(
                "❌ Unable to process this audio file."
            )

            with st.expander(
                "Technical error details"
            ):

                st.exception(error)

        finally:

            # ------------------------------------------------
            # CLEAN UP TEMPORARY AUDIO
            # ------------------------------------------------

            if temp_path is not None:

                try:

                    if temp_path.exists():
                        temp_path.unlink()

                except OSError:
                    pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Age & Emotion Detection through Voice • '
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
