import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Big Cat Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Hide Streamlit UI Chrome ─────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide watermark, menu, footer, toolbar */
    #MainMenu                        { visibility: hidden; }
    footer                           { visibility: hidden; }
    [data-testid="stToolbar"]        { visibility: hidden; }
    [data-testid="stDecoration"]     { display: none; }
    [data-testid="stStatusWidget"]   { visibility: hidden; }
    header[data-testid="stHeader"]   { background: transparent; }

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #1a1c20;
        font-family: 'Georgia', serif;
    }
    [data-testid="block-container"] {
        padding-top: 1.5rem;
        max-width: 720px;
    }

    /* ── Header ── */
    .header-wrap {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .header-eyebrow {
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #c8a96e;
        margin-bottom: 0.5rem;
    }
    .header-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #f0ece3;
        letter-spacing: -0.01em;
        line-height: 1.1;
        margin: 0;
    }
    .header-sub {
        font-size: 0.88rem;
        color: #5a5a5a;
        margin-top: 0.6rem;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.05em;
    }
    .header-rule {
        border: none;
        border-top: 1px solid #2e2e2e;
        margin: 1.5rem 0;
    }

    /* ── Drop Zone ── */
    .drop-label {
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #5a5a5a;
        margin-bottom: 0.5rem;
    }
    [data-testid="stFileUploader"] > div {
        border: 1.5px dashed #3a3a2a !important;
        border-radius: 6px !important;
        background: #1e2024 !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #c8a96e !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        padding: 2rem !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #4a4a4a !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Image Card ── */
    .img-wrap {
        background: #1e2024;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .img-tag {
        font-family: 'Courier New', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #3e3e3e;
        margin-bottom: 0.5rem;
    }

    /* ── Verdict ── */
    .verdict-wrap {
        border-radius: 4px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .verdict-identified { background: #111a10; border-color: #4d9e3f; }
    .verdict-unknown    { background: #1a1510; border-color: #c8a96e; }

    .verdict-eyebrow {
        font-family: 'Courier New', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.3rem;
    }
    .verdict-name {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        line-height: 1.1;
    }
    .verdict-identified .verdict-name { color: #6ecf5e; }
    .verdict-unknown    .verdict-name { color: #c8a96e; }

    .verdict-conf {
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .verdict-identified .verdict-conf { color: #2d6b24; }
    .verdict-unknown    .verdict-conf { color: #7a6030; }

    /* ── Probability Bars ── */
    .prob-section {
        margin-top: 0.5rem;
    }
    .prob-heading {
        font-family: 'Courier New', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #3e3e3e;
        margin-bottom: 0.8rem;
    }
    .prob-row { margin-bottom: 0.85rem; }
    .prob-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    .prob-name {
        font-family: 'Courier New', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: #888;
    }
    .prob-pct {
        font-family: 'Courier New', monospace;
        font-size: 0.72rem;
        color: #555;
    }
    .prob-track {
        height: 2px;
        background: #222;
        border-radius: 1px;
        overflow: hidden;
    }
    .prob-fill {
        height: 2px;
        border-radius: 1px;
        background: #c8a96e;
        transition: width 0.4s ease;
    }
    .prob-fill-top { background: #6ecf5e; }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        color: #2e2e2e;
        border-top: 1px solid #222;
        margin-top: 2rem;
        text-transform: uppercase;
    }
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
<div class="header-wrap">
    <div class="header-eyebrow">Deep Learning &nbsp;·&nbsp; Computer Vision</div>
    <div class="header-title">Big Cat Classifier</div>
    <div class="header-sub">Tiger &nbsp;·&nbsp; Clouded Leopard &nbsp;·&nbsp; Snow Leopard &nbsp;·&nbsp; African Leopard</div>
</div>
<hr class="header-rule">
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────

st.markdown('<div class="drop-label">Drop an image or click to browse</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ─── Inference ────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Image preview
    st.markdown('<div class="img-wrap"><div class="img-tag">Input Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img_array = np.array(image.resize(IMG_SIZE)).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Running inference..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence      = predictions[0][predicted_index] * 100
    predicted_class = CLASS_NAMES[predicted_index]

    # Verdict
    if confidence >= THRESHOLD * 100:
        st.markdown(f"""
        <div class="verdict-wrap verdict-identified">
            <div>
                <div class="verdict-eyebrow">Identified Species</div>
                <div class="verdict-name">{predicted_class}</div>
            </div>
            <div class="verdict-conf">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-wrap verdict-unknown">
            <div>
                <div class="verdict-eyebrow">Low Confidence</div>
                <div class="verdict-name">Not a Leopard or Tiger</div>
            </div>
            <div class="verdict-conf">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown('<div class="prob-section"><div class="prob-heading">Confidence breakdown</div>', unsafe_allow_html=True)

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
        pct      = prob * 100
        is_top   = (i == predicted_index)
        fill_cls = "prob-fill prob-fill-top" if is_top else "prob-fill"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-meta">
                <span class="prob-name">{name}</span>
                <span class="prob-pct">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="{fill_cls}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;font-family:'Courier New',monospace;
                font-size:0.75rem;letter-spacing:0.15em;color:#2a2a2a;text-transform:uppercase;">
        No image loaded — drag and drop or click above
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    Big Cat Classifier &nbsp;·&nbsp; MobileNetV3 Transfer Learning &nbsp;·&nbsp; TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
