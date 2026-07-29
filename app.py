import os
import streamlit as st
import tensorflow as st_tf
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Big Cat Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Reset Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="block-container"]  { padding-top: 2rem; max-width: 680px; }

/* ── Page base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f6f8fa;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #24292f;
}

/* ── Header ── */
.gh-header {
    padding: 2rem 0 1.2rem;
    border-bottom: 1px solid #d0d7de;
    margin-bottom: 1.6rem;
}
.gh-header-top {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.3rem;
}
.gh-icon {
    width: 28px; height: 28px;
    background: #24292f;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; color: #f6f8fa; font-weight: 700;
    flex-shrink: 0;
}
.gh-repo-path {
    font-size: 1.1rem;
    color: #57606a;
    font-weight: 400;
}
.gh-repo-path span {
    color: #0969da;
    font-weight: 600;
}
.gh-desc {
    font-size: 0.82rem;
    color: #57606a;
    margin-top: 0.2rem;
    padding-left: 2.5rem;
}
.gh-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 2em;
    font-size: 0.7rem;
    font-weight: 600;
    background: #ddf4ff;
    color: #0969da;
    border: 1px solid #b6e3ff;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* ── GitHub-style drop zone ── */
.gh-drop-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #24292f;
    margin-bottom: 0.5rem;
}
.gh-drop-hint {
    font-size: 0.72rem;
    color: #57606a;
    margin-bottom: 0.6rem;
}

/* Hide the default Streamlit uploader and overlay our own zone on top */
[data-testid="stFileUploader"] {
    position: relative;
}
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 2px dashed #d0d7de !important;
    border-radius: 6px !important;
    padding: 2.5rem 1.5rem !important;
    text-align: center !important;
    transition: border-color 0.15s, background 0.15s !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0969da !important;
    background: #f0f6ff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    font-size: 0.82rem !important;
    color: #57606a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span:first-child {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #0969da !important;
}
/* Browse button */
[data-testid="stFileUploaderDropzone"] button {
    background: #f6f8fa !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px !important;
    color: #24292f !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.9rem !important;
    margin-top: 0.6rem !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #eaeef2 !important;
}

/* ── Image preview card (GitHub attachment style) ── */
.preview-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}
.preview-card-header {
    background: #f6f8fa;
    border-bottom: 1px solid #d0d7de;
    padding: 0.5rem 0.9rem;
    font-size: 0.75rem;
    color: #57606a;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.preview-card-body { padding: 0.75rem; }

/* ── Result card ── */
.result-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}
.result-card-header {
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    border-bottom: 1px solid #d0d7de;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-identified { background: #dafbe1; color: #116329; border-color: #aceebb; }
.result-unknown    { background: #fff8c5; color: #7d4e00; border-color: #e3b341; }

.result-card-body { padding: 1rem; }

.result-species {
    font-size: 1.5rem;
    font-weight: 700;
    color: #24292f;
    margin-bottom: 0.2rem;
}
.result-conf-line {
    font-size: 0.8rem;
    color: #57606a;
}
.result-conf-val {
    font-weight: 600;
    color: #0969da;
}

/* ── Probability section ── */
.prob-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #57606a;
    margin: 1rem 0 0.7rem;
    padding-top: 0.8rem;
    border-top: 1px solid #d0d7de;
}
.prob-row { margin-bottom: 0.7rem; }
.prob-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
}
.prob-name { font-size: 0.76rem; color: #24292f; font-weight: 500; }
.prob-pct  { font-size: 0.76rem; color: #57606a; }
.prob-track {
    height: 6px;
    background: #eaeef2;
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill     { height: 6px; border-radius: 3px; background: #0969da; }
.prob-fill-top { height: 6px; border-radius: 3px; background: #1a7f37; }
.prob-fill-low { height: 6px; border-radius: 3px; background: #9a6700; }

/* ── Empty state ── */
.empty-state {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-state-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.empty-state-text { font-size: 0.85rem; color: #57606a; }
.empty-state-sub  { font-size: 0.75rem; color: #8c959f; margin-top: 0.3rem; }

/* ── Footer ── */
.gh-footer {
    border-top: 1px solid #d0d7de;
    padding: 1.2rem 0;
    margin-top: 2rem;
    font-size: 0.72rem;
    color: #8c959f;
    text-align: center;
}
.gh-footer a { color: #0969da; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ['TIGER', 'CLOUDED LEOPARD', 'SNOW LEOPARD', 'AFRICAN LEOPARD']
IMG_SIZE    = (224, 224)
MODEL_PATH  = "models/tl_feature_extraction_best.keras"
THRESHOLD   = 0.80

# ─── Load Model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="gh-header">
    <div class="gh-header-top">
        <div class="gh-icon">BC</div>
        <div class="gh-repo-path">
            RightFix / <span>big-cat-classifier</span>
            <span class="gh-badge">Public</span>
        </div>
    </div>
    <div class="gh-desc">
        Transfer learning classifier for big cat species &nbsp;&middot;&nbsp;
        MobileNetV3 &nbsp;&middot;&nbsp; 4 species &nbsp;&middot;&nbsp; 98% accuracy
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Drop Zone ────────────────────────────────────────────────────────────────

st.markdown('<div class="gh-drop-label">Upload a cat image to classify</div>', unsafe_allow_html=True)
st.markdown('<div class="gh-drop-hint">Drag and drop your file here, or click to browse. Supports JPG and PNG.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ─── Inference ────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Image preview card
    st.markdown(f"""
    <div class="preview-card">
        <div class="preview-card-header">
            &#128247; &nbsp; {uploaded_file.name}
        </div>
        <div class="preview-card-body">
    """, unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Preprocess + predict
    img_array = np.array(image.resize(IMG_SIZE)).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_index] * 100
    predicted_class = CLASS_NAMES[predicted_index]
    identified      = confidence >= THRESHOLD * 100

    # Result card
    if identified:
        header_cls  = "result-identified"
        header_text = "Species identified"
        species_out = predicted_class
    else:
        header_cls  = "result-unknown"
        header_text = "Low confidence — unable to identify"
        species_out = "Not a Leopard or Tiger"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-header {header_cls}">
            <span>{header_text}</span>
            <span>{confidence:.1f}% confidence</span>
        </div>
        <div class="result-card-body">
            <div class="result-species">{species_out}</div>
            <div class="result-conf-line">
                Model confidence: <span class="result-conf-val">{confidence:.2f}%</span>
                &nbsp;&middot;&nbsp; Threshold: {int(THRESHOLD*100)}%
            </div>
            <div class="prob-title">All class probabilities</div>
    """, unsafe_allow_html=True)

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
        pct = prob * 100
        if i == predicted_index and identified:
            fill = "prob-fill-top"
        elif i == predicted_index:
            fill = "prob-fill-low"
        else:
            fill = "prob-fill"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-meta">
                <span class="prob-name">{name}</span>
                <span class="prob-pct">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="{fill}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">&#128247;</div>
        <div class="empty-state-text">No image uploaded yet</div>
        <div class="empty-state-sub">Drop an image above to run the classifier</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="gh-footer">
    Big Cat Classifier &nbsp;&middot;&nbsp;
    MobileNetV3 Transfer Learning &nbsp;&middot;&nbsp;
    GET 324 Group C10 &nbsp;&middot;&nbsp;
    <a href="https://github.com/RightFix/big-cat-classifier" target="_blank">View on GitHub</a>
</div>
""", unsafe_allow_html=True)
