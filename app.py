import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef4ff, #ffffff);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.section-title {
    color: #1e293b;
    font-weight: bold;
    margin-bottom: 15px;
}

.prediction-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-radius: 20px;
    padding: 25px;
    border-left: 8px solid #2563eb;
}

.high {
    color: red;
    font-weight: bold;
}

.medium {
    color: orange;
    font-weight: bold;
}

.low {
    color: green;
    font-weight: bold;
}

.recommend-box {
    background: #fff8e7;
    border-left: 8px solid orange;
    border-radius: 15px;
    padding: 20px;
}

.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 15px;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL FROM GOOGLE DRIVE
# ==========================================================

import os
import gdown

MODEL_PATH = "road_damage_model.h5"

FILE_ID = "1wJ0u_JzWKbS5o-DBd-9T-TLmujnb3ZLG"

class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_model():

    # download only once
    if not os.path.exists(MODEL_PATH):

        url = (
            f"https://drive.google.com/uc?id={FILE_ID}"
        )

        with st.spinner(
            "Downloading CNN model..."
        ):
            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'Dense': CustomDense}
    )

    return model


model = load_model()
# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

# ==========================================================
# SECTION 1 — HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '🛣️ AI-Based Road Damage Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart City Infrastructure Monitoring using CNN'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# SECTION 2 — ABOUT PROJECT
# ==========================================================

st.markdown("""
<div class="card">

<h2 class="section-title">
📌 About the Project
</h2>

<h4>Why Road Monitoring is Important?</h4>

Road damage monitoring is essential to improve public
safety, reduce accidents, and support efficient
transportation systems. Early identification of road
damage helps governments and municipalities reduce
maintenance costs and improve road quality.

<h4>Role of CNN in Computer Vision</h4>

Convolutional Neural Networks (CNNs) are powerful deep
learning models used in computer vision to automatically
extract image features and classify patterns such as
potholes, cracks, and road defects.

<h4>Practical Industry Applications</h4>

• Smart city monitoring systems  
• Autonomous vehicle navigation  
• Road infrastructure inspection  
• Transportation safety analytics  
• Government maintenance planning

</div>
""", unsafe_allow_html=True)

# ==========================================================
# SECTION 3 — UPLOAD AREA
# ==========================================================

st.markdown("""
<div class="card">
<h2 class="section-title">
📤 Upload Road Image
</h2>
Upload a road image for damage detection.
Supports drag-and-drop upload.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# MAIN WORKFLOW
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # ======================================================
    # SECTION 4 — IMAGE PREVIEW
    # ======================================================

    with col1:

        st.markdown("""
        <div class="card">
        <h2 class="section-title">
        🖼 Uploaded Image Preview
        </h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # ======================================================
    # PREDICTION
    # ======================================================

    with col2:

        if st.button("🔍 Detect Damage"):

            img = image.resize((128,128))

            img_array = np.array(img)

            if len(img_array.shape) == 2:
                img_array = np.stack(
                    [img_array]*3,
                    axis=-1
                )

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(
                img_array,
                verbose=0
            )[0]

            pred_index = np.argmax(
                prediction
            )

            confidence = (
                prediction[pred_index] * 100
            )

            predicted_label = class_names[
                pred_index
            ]

            # ==============================================
            # Severity Logic
            # ==============================================

            if confidence >= 85:
                severity = "High"
                sev_class = "high"

            elif confidence >= 60:
                severity = "Medium"
                sev_class = "medium"

            else:
                severity = "Low"
                sev_class = "low"

            # ==============================================
            # SECTION 5 — PREDICTION AREA
            # ==============================================

            st.markdown(f"""
            <div class="prediction-card">

            <h2 class="section-title">
            🤖 Prediction Result
            </h2>

            <h3>
            Prediction:
            {predicted_label} Detected
            </h3>

            <h3>
            Confidence:
            {confidence:.2f}%
            </h3>

            <h3 class="{sev_class}">
            Severity:
            {severity}
            </h3>

            </div>
            """, unsafe_allow_html=True)

            # ==============================================
            # METRICS
            # ==============================================

            st.markdown("### 📊 Prediction Measures")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Prediction",
                    predicted_label
                )

            with m2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with m3:
                st.metric(
                    "Severity",
                    severity
                )

            # ==============================================
            # SECTION 6 — VISUALIZATION AREA
            # ==============================================

            st.markdown("""
            ## 📈 Visualization Area
            """)

            fig, ax = plt.subplots(
                figsize=(8,4)
            )

            labels = [
                "Pothole",
                "Crack",
                "Manhole"
            ]

            ax.bar(
                labels,
                prediction
            )

            ax.set_ylim([0,1])
            ax.set_ylabel(
                "Confidence"
            )

            ax.set_title(
                "Class Confidence Graph"
            )

            st.pyplot(fig)

            # ==============================================
            # SECTION 7 — RECOMMENDATIONS
            # ==============================================

            st.markdown("""
            <div class="recommend-box">
            <h2>
            🚧 Recommendations
            </h2>
            """, unsafe_allow_html=True)

            if severity == "High":

                st.error("""
Immediate maintenance recommended.

⚠ High-risk road condition detected.
Public safety may be affected.
""")

            elif severity == "Medium":

                st.warning("""
Moderate repair priority suggested.

Road inspection recommended.
""")

            else:

                st.success("""
Low-risk condition detected.

Routine maintenance is sufficient.
""")

            st.markdown("</div>",
                        unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<center>

AI-Based Road Damage Detection System  
Built using CNN + TensorFlow + Streamlit

</center>
""", unsafe_allow_html=True)