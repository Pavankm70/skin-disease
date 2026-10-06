import os
from dotenv import load_dotenv
from google import genai
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Skin Lesion Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(197, 229, 218, 0.34), transparent 34%),
        linear-gradient(180deg, #f4f8f6 0%, #f8faf9 48%, #f2f7f5 100%);
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3.5rem;
    max-width: 1180px;
}

.main-title {
    color: #153e38;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.045em;
    line-height: 1.12;
    margin: 0.4rem 0 0.55rem;
}

.subtitle {
    color: #52736b;
    text-align: center;
    font-size: 1.05rem;
    margin: 0 auto 1.7rem;
    max-width: 700px;
}

.section-text {
    color: #52736b;
    font-size: 0.98rem;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: #678078;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 2.5rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(37, 111, 94, 0.14);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 10px 28px rgba(23, 65, 55, 0.07);
}

[data-testid="stMetric"] {
    border: 1px solid rgba(37, 111, 94, 0.13);
    border-radius: 14px;
    background: linear-gradient(145deg, #ffffff, #f2f8f5);
    padding: 1rem 1.1rem;
    box-shadow: 0 6px 18px rgba(23, 65, 55, 0.06);
}

[data-testid="stFileUploader"] section {
    border: 1.5px dashed #8db9aa;
    border-radius: 16px;
    background: rgba(237, 247, 242, 0.72);
    transition: border-color 160ms ease, background 160ms ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #27836d;
    background: #e8f4ee;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 650;
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 16px rgba(28, 112, 90, 0.18);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf5f1 0%, #e8f1ed 100%);
    border-right: 1px solid rgba(37, 111, 94, 0.12);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    gemini_client = None


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_skin_model():

    model_path = Path(__file__).resolve().parent / "best_skin_model_phase2.keras"

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model

model = load_skin_model()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


CLASS_FULL_NAMES = {

    "akiec": "Actinic Keratoses",

    "bcc": "Basal Cell Carcinoma",

    "bkl": "Benign Keratosis-like Lesions",

    "df": "Dermatofibroma",

    "mel": "Melanoma",

    "nv": "Melanocytic Nevi",

    "vasc": "Vascular Lesions"
}


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None


if "full_name" not in st.session_state:
    st.session_state.full_name = None


if "confidence" not in st.session_state:
    st.session_state.confidence = None


if "gemini_explanation" not in st.session_state:
    st.session_state.gemini_explanation = None


if "gemini_answer" not in st.session_state:
    st.session_state.gemini_answer = None


if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"


if "previous_language" not in st.session_state:
    st.session_state.previous_language = "English"


if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔬 System Information")

    st.caption(
        "AI-powered skin lesion classification "
        "with generative AI assistance."
    )

    st.divider()


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("🧠 Deep Learning Model")

    st.write("**Framework:** TensorFlow")

    st.write("**Input:** 300 × 300 RGB")

    st.write("**Output Classes:** 7")


    st.divider()


    # --------------------------------------------------------
    # LESION CLASSES
    # --------------------------------------------------------

    st.subheader("🏷️ Lesion Classes")

    st.markdown("""
- Actinic Keratoses
- Basal Cell Carcinoma
- Benign Keratosis-like Lesions
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Lesions
""")


    st.divider()


    # --------------------------------------------------------
    # GEMINI INFORMATION
    # --------------------------------------------------------

    st.subheader("✨ Generative AI")

    st.write("**Assistant:** Google Gemini")

    st.write(
        "**Languages:** English, Kannada, Hindi"
    )

    st.caption(
        "Gemini provides educational explanations "
        "and answers follow-up questions based on "
        "the model prediction."
    )


    st.divider()


    # --------------------------------------------------------
    # GEMINI STATUS
    # --------------------------------------------------------

    if gemini_client is not None:

        st.success("Gemini Connected")

    else:

        st.error("Gemini Not Connected")


    st.warning(
        "Educational and research use only. "
        "This system is not a medical diagnostic tool."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔬 AI Skin Lesion Analyzer'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Deep Learning Classification + '
    'Multilingual Gemini AI Assistant'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Upload a skin lesion image to obtain an AI-generated "
    "classification and educational information about the "
    "predicted lesion category."
)

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.badge("Image classification", icon=":material/center_focus_strong:", color="green")
with status_col2:
    st.badge("7 lesion categories", icon=":material/category:", color="blue")
with status_col3:
    st.badge(
        "Gemini assistant ready" if gemini_client is not None else "Gemini key not configured",
        icon=":material/auto_awesome:",
        color="violet" if gemini_client is not None else "gray",
    )


st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Skin Lesion Analysis")


st.markdown(
    '<div class="section-text">'
    'Upload a JPG, JPEG, or PNG skin lesion image for analysis.'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)
st.caption("Supported formats: JPG, JPEG, PNG · The image is resized for the model during analysis.")


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:


    # ========================================================
    # RESET WHEN NEW IMAGE IS UPLOADED
    # ========================================================

    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        st.session_state.prediction_done = False

        st.session_state.predicted_class = None

        st.session_state.full_name = None

        st.session_state.confidence = None

        st.session_state.gemini_explanation = None

        st.session_state.gemini_answer = None

        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    image = Image.open(uploaded_file)


    st.success(
        "✅ Image uploaded successfully."
    )


    image_col, action_col = st.columns(
        [1, 1.3],
        gap="large"
    )


    # ========================================================
    # LEFT COLUMN - IMAGE
    # ========================================================

    with image_col:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch",
            alt="Uploaded skin lesion image preview",
        )


    # ========================================================
    # RIGHT COLUMN - CLASSIFICATION
    # ========================================================

    with action_col:

        st.subheader(
            "AI Classification"
        )

        st.write(
            "Click the button below to analyze the uploaded "
            "image using the trained deep learning model."
        )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing skin lesion image..."
            ):


                # ---------------------------------------------
                # IMAGE PREPROCESSING
                # ---------------------------------------------

                processed_image = image.convert(
                    "RGB"
                )


                processed_image = (
                    processed_image.resize(
                        (300, 300)
                    )
                )


                img_array = np.array(
                    processed_image
                )


                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # =============================================
                # MODEL PREDICTION
                # =============================================

                predictions = model.predict(
                    img_array,
                    verbose=0
                )


                predicted_index = np.argmax(
                    predictions[0]
                )


                predicted_class = CLASS_NAMES[
                    predicted_index
                ]


                confidence = float(
                    predictions[0][
                        predicted_index
                    ]
                )


                full_name = CLASS_FULL_NAMES[
                    predicted_class
                ]


                # =============================================
                # SAVE RESULT
                # =============================================

                st.session_state.prediction_done = True


                st.session_state.predicted_class = (
                    predicted_class
                )


                st.session_state.full_name = (
                    full_name
                )


                st.session_state.confidence = (
                    confidence
                )


                # Clear previous Gemini responses
                st.session_state.gemini_explanation = None

                st.session_state.gemini_answer = None


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if st.session_state.prediction_done:


            predicted_class = (
                st.session_state.predicted_class
            )


            full_name = (
                st.session_state.full_name
            )


            confidence = (
                st.session_state.confidence
            )


            st.divider()


            st.subheader(
                "🧠 Analysis Result"
            )


            result_col1, result_col2 = (
                st.columns(2)
            )


            with result_col1:

                st.metric(
                    label="Predicted Lesion",
                    value=full_name
                )


            with result_col2:

                st.metric(
                    label="Confidence",
                    value=(
                        f"{confidence * 100:.2f}%"
                    )
                )

            st.progress(
                confidence,
                text=f"Model confidence · {confidence * 100:.1f}%",
            )


            if confidence < 0.60:
                st.warning(
                    "⚠️ Low-confidence prediction. "
                    "The uploaded image does not strongly match "
                    "any of the trained skin-lesion categories."
                )

                st.info(
                    "This model is trained to classify 7 specific "
                    "skin-lesion categories and does not include a "
                    "separate normal-skin class."
                )

            else:
                st.success(
                    f"Predicted category: "
                    f"**{full_name} "
                    f"({predicted_class})**"
                )
            


    # ========================================================
    # GEMINI SECTION
    # ========================================================

    if st.session_state.prediction_done:


        predicted_class = (
            st.session_state.predicted_class
        )


        full_name = (
            st.session_state.full_name
        )


        confidence = (
            st.session_state.confidence
        )


        st.divider()


        st.warning(
            "⚠️ The classification above is generated by an "
            "AI model and must not be considered a confirmed "
            "medical diagnosis."
        )


        # ====================================================
        # LANGUAGE SELECTION
        # ====================================================

        st.header(
            "🌐 Language"
        )


        selected_language = st.selectbox(
            "Choose the language for Gemini responses",
            [
                "English",
                "Kannada",
                "Hindi"
            ],
            key="selected_language"
        )


        # ----------------------------------------------------
        # CLEAR OLD RESPONSE WHEN LANGUAGE CHANGES
        # ----------------------------------------------------

        if (
            st.session_state.previous_language
            != selected_language
        ):

            st.session_state.gemini_explanation = None

            st.session_state.gemini_answer = None

            st.session_state.previous_language = (
                selected_language
            )


        # ====================================================
        # GEMINI EXPLANATION
        # ====================================================

        st.header(
            "✨ AI Educational Explanation"
        )


        st.write(
            "Generate a concise educational overview of "
            f"**{full_name}** in "
            f"**{selected_language}** using Gemini."
        )


        if gemini_client is not None:


            if st.button(
                "✨ Generate AI Explanation"
            ):


                with st.spinner(
                    f"Gemini is generating an explanation "
                    f"in {selected_language}..."
                ):


                    try:


                        prompt = f"""
You are an educational AI assistant.

LANGUAGE INSTRUCTION:

Respond entirely in {selected_language}.

Use simple, natural and easy-to-understand
{selected_language}.

If the selected language is Kannada,
write the response using Kannada script.

If the selected language is Hindi,
write the response using Devanagari Hindi script.

Medical terminology may include the standard English
medical term in parentheses when helpful.

A skin lesion classification model produced the
following prediction:

Predicted lesion category: {full_name}

Dataset class: {predicted_class}

Model confidence: {confidence * 100:.2f}%

Provide a SHORT educational explanation of this
predicted skin lesion category.

Use the following structure:

### What is it?

Explain it in 2-3 simple sentences.

### Common Characteristics

Give 3-4 short bullet points.

### General Risk

Explain the general risk in 2-3 sentences.

### When to Seek Medical Advice

Give 3-4 short bullet points.

Keep the entire response concise and easy to understand.

IMPORTANT:

This result comes from an AI image classification model.

Do not say that the user definitely has this condition.

Do not provide a confirmed medical diagnosis.

Do not prescribe medication or treatment.

Clearly state that the information is educational
and does not replace evaluation by a qualified
healthcare professional.
"""


                        response = (
                            gemini_client.models.generate_content(
                                model="gemini-3.5-flash-lite",
                                contents=prompt
                            )
                        )


                        st.session_state.gemini_explanation = (
                            response.text
                        )


                    except Exception as e:


                        st.error(
                            "❌ Gemini could not generate "
                            "the explanation."
                        )


                        st.exception(e)


            # =================================================
            # DISPLAY EXPLANATION
            # =================================================

            if st.session_state.gemini_explanation:


                st.success(
                    f"✅ AI explanation generated "
                    f"successfully in "
                    f"{selected_language}."
                )


                with st.container(
                    border=True
                ):


                    st.markdown(
                        st.session_state.gemini_explanation
                    )


            # =================================================
            # ASK GEMINI
            # =================================================

            st.divider()


            st.header(
                "💬 Ask Gemini"
            )


            st.write(
                "Ask an educational follow-up question about "
                f"**{full_name}**. Gemini will answer in "
                f"**{selected_language}**."
            )


            # =================================================
            # LANGUAGE SPECIFIC PLACEHOLDER
            # =================================================

            if selected_language == "Kannada":

                question_placeholder = (
                    "ಉದಾಹರಣೆ: ಇದು ಮೆಲನೋಮಾದಿಂದ "
                    "ಹೇಗೆ ಭಿನ್ನವಾಗಿದೆ?"
                )


            elif selected_language == "Hindi":

                question_placeholder = (
                    "उदाहरण: यह मेलानोमा से "
                    "कैसे अलग है?"
                )


            else:

                question_placeholder = (
                    "Example: How is this different "
                    "from melanoma?"
                )


            # =================================================
            # QUESTION INPUT
            # =================================================

            user_question = st.text_input(
                "Your Question",
                placeholder=question_placeholder,
                key="user_question"
            )


            # =================================================
            # ASK BUTTON
            # =================================================

            if st.button(
                "💬 Ask Gemini",
                type="secondary"
            ):


                if user_question.strip():


                    with st.spinner(
                        f"Gemini is answering in "
                        f"{selected_language}..."
                    ):


                        try:


                            question_prompt = f"""
You are an educational AI assistant integrated
into a skin lesion classification application.

LANGUAGE INSTRUCTION:

The selected response language is {selected_language}.

Answer entirely in {selected_language}.

Use simple, natural and easy-to-understand
{selected_language}.

If the selected language is Kannada,
write the answer using Kannada script.

If the selected language is Hindi,
write the answer using Devanagari Hindi script.

The user may ask the question in English,
Kannada or Hindi.

Understand the user's question regardless of whether
it is written in English, Kannada or Hindi.

However, ALWAYS respond in {selected_language}.

Medical terminology may include the standard English
medical term in parentheses when helpful.

The image classification model produced:

Predicted lesion category: {full_name}

Dataset class: {predicted_class}

Model confidence: {confidence * 100:.2f}%

The user asks:

"{user_question}"

Answer the user's question clearly and concisely.

Keep the answer focused on the question.

IMPORTANT RULES:

- The model result is only an AI prediction.

- Do not state that the user definitely has this condition.

- Do not provide a confirmed medical diagnosis.

- Do not prescribe medication or treatment.

- Do not provide medication dosages.

- If the question requires personal medical assessment,
  explain that a qualified healthcare professional should
  be consulted.

- Keep the response educational.
"""


                            answer_response = (
                                gemini_client.models.generate_content(
                                    model="gemini-3.6-flash",
                                    contents=question_prompt
                                )
                            )


                            st.session_state.gemini_answer = (
                                answer_response.text
                            )


                        except Exception as e:


                            st.error(
                                "❌ Gemini could not answer "
                                "the question."
                            )


                            st.exception(e)


                else:


                    st.warning(
                        "⚠️ Please enter a question first."
                    )


            # =================================================
            # DISPLAY GEMINI ANSWER
            # =================================================

            if st.session_state.gemini_answer:


                with st.container(
                    border=True
                ):


                    st.markdown(
                        "### 🤖 Gemini Answer"
                    )


                    st.markdown(
                        st.session_state.gemini_answer
                    )


        else:


            st.error(
                "❌ Gemini API key was not found. "
                "Check your .env file."
            )


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:


    st.info(
        "👆 Upload a skin lesion image above "
        "to begin analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div class="footer">
        🔬 AI Skin Lesion Analyzer<br>
        Deep Learning Classification +
        Multilingual Generative AI Assistance<br><br>

        🌐 English • ಕನ್ನಡ • हिन्दी<br><br>

        ⚠️ For educational and research purposes only.
        Not intended for medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
