import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Big Cat Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS Custom Styling ───────────────────────────────────────────────────────

st.markdown("""
<style>
/* Reset Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { 
    display: none !important; 
}
header[data-testid="stHeader"] { 
    background: transparent; 
}
[data-testid="block-container"] { 
    padding-top: 2rem; 
    max-width: 680px; 
}

/* Base Page Styling */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

/* Clean Header */
.app-header {
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}
.app-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
}
.app-desc {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.35rem;
}

/* ── Integrated "No Image Provided" Upload Card ── */
[data-testid="stFileUploader"] {
    position: relative;
    margin-bottom: 1.5rem;
}

/* Hide default buttons, icons, and size limit text inside the dropzone */
[data-testid="stFileUploaderInstructions"] small,
[data-testid="stFileUploaderLimitData"],
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] svg {
    display: none !important;
}

/* Convert the uploader container into the "No Image Provided" card */
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 3.5rem 1.5rem !important;
    text-align: center !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
}

/* Format text inside the dropzone to show "No image provided" UI */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "No image provided";
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Drag and drop an image here, or click to upload";
    font-size: 0.85rem !important;
    color: #64748b !important;
}

/* Cards UI */
.custom-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.badge-success {
    background-color: #dcfce7;
    color: #15803d;
}
.badge-warning {
    background-color: #fef9c3;
    color: #a16207;
}

.species-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
.conf-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.25rem;
}

/* Custom Progress Bars */
.prob-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.prob-track {
    height: 8px;
    background-color: #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.85rem;
}
.prob-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Footer */
.app-footer {
    border-top: 1px solid #e2e8f0;
    padding: 1.5rem 0;
    margin-top: 2.5rem;
    font-size: 0.75rem;
    color: #94a3b8;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ['TIGER', 'CLOUDED LEOPARD', 'SNOW LEOPARD', 'AFRICAN LEOPARD']
IMG_SIZE    = (224, 224)
MODEL_PATH  = "models/tl_feature_extraction_best.keras"
THRESHOLD   = 0.70

# ─── Load Model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at path: `{MODEL_PATH}`")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ─── Header Section ───────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="app-title">Big Cat Classifier</div>
    <div class="app-desc">
        Deep learning transfer model (MobileNetV3) tuned for distinguishing between Tigers and Leopard species.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── "No Image Provided" Integrated Upload Area ──────────────────────────────

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

# ─── Prediction Logic & Visualisation ─────────────────────────────────────────

if uploaded_file is not None:
    # 1. Image Preview Box
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("##### Selected Image")
    st.image(image, use_container_width=True)

    # 2. Preprocess Image
    img_array = np.array(image.resize(IMG_SIZE)).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Model Inference
    with st.spinner("Analyzing visual features..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_index] * 100
    predicted_class = CLASS_NAMES[predicted_index]
    identified      = (confidence / 100.0) >= THRESHOLD

    # 4. Result Display Card
    badge_class = "badge-success" if identified else "badge-warning"
    badge_text  = "SPECIES IDENTIFIED" if identified else "LOW CONFIDENCE"
    display_title = predicted_class if identified else "Uncertain / Unrecognized"

    st.markdown(f"""
    <div class="custom-card">
        <span class="status-badge {badge_class}">{badge_text}</span>
        <div class="species-title">{display_title}</div>
        <div class="conf-subtitle">
            Highest match confidence: <strong>{confidence:.2f}%</strong> &nbsp;&bull;&nbsp; Required threshold: {int(THRESHOLD*100)}%
        </div>
        <hr style="border: none; border-top: 1px solid #f1f5f9; margin: 1rem 0;" />
        <div style="font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.75rem;">
            Class Probabilities
        </div>
    """, unsafe_allow_html=True)

    # Render probability bars dynamically
    for name, prob in zip(CLASS_NAMES, predictions[0]):
        pct = prob * 100
        is_top = (name == predicted_class)
        
        if is_top and identified:
            bar_color = "#2563eb"
        elif is_top:
            bar_color = "#eab308"
        else:
            bar_color = "#cbd5e1"

        st.markdown(f"""
        <div>
            <div class="prob-meta">
                <span style="color: {'#0f172a' if is_top else '#64748b'};">{name}</span>
                <span style="color: {'#2563eb' if is_top else '#64748b'};">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: {pct:.1f}%; background-color: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── Clean App Footer ─────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    Big Cat Classifier &nbsp;&bull;&nbsp; MobileNetV3 Neural Network &nbsp;&bull;&nbsp; Multi-Class Inference Engine
</div>
""", unsafe_allow_html=True)
