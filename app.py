import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Big Cat Classifier",
    layout="centered"
)

CLASS_NAMES = ['TIGER', 'CLOUDED LEOPARD', 'SNOW LEOPARD', 'AFRICAN LEOPARD']  
IMG_SIZE   = (224, 224)
MODEL_PATH = "models/tl_feature_extraction_best.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Big Cat Classifier")
st.markdown("Upload an image of a big cat and the model will identify it.")
st.divider()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image.resize(IMG_SIZE)).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = predictions[0][predicted_index] * 100

    with col2:
        st.markdown("### Result")
        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.markdown("### All Probabilities")
        for name, prob in zip(CLASS_NAMES, predictions[0]):
            st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

st.divider()
st.caption("Big Cat Classifier | TensorFlow & Streamlit")
