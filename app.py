
import streamlit as st
import os
import numpy as np
import rasterio
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils.predictor import predict_tiff, classify_prediction, show_images
from tempfile import NamedTemporaryFile
import gdown

# App Title
st.set_page_config(layout="wide")
st.title("🌆 Future Prediction: Urban Expansion")

# Load Model from Google Drive
@st.cache_resource
def load_model():
    model_path = "model/model.h5"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        url = "https://drive.google.com/uc?id=1vkeZmAIzop8K5MdsIK8o70xpEHN7YVH6"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()
st.success("✅ Model Loaded Successfully")

# File uploader
uploaded_files = st.file_uploader("📂 Upload TIFF files (at least 2)", type=["tif", "tiff"], accept_multiple_files=True)

# Year input
target_year = st.number_input("📅 Enter the Target Year for Prediction", min_value=2020, max_value=2100, step=1)

if uploaded_files and len(uploaded_files) >= 2 and target_year:
    st.info(f"🛠️ Running prediction for the year {target_year}...")

    for i, uploaded_file in enumerate(uploaded_files):
        with NamedTemporaryFile(delete=False, suffix=".tif") as input_tmp:
            input_tmp.write(uploaded_file.read())
            input_tmp_path = input_tmp.name

        output_tmp_path = input_tmp_path.replace(".tif", f"_predicted_{target_year}.tif")

        original, predicted = predict_tiff(input_tmp_path, output_tmp_path, model)

        # Apply binary classification: 1 = Built-up, 0 = Non Built-up
        binary_predicted = (predicted >= 0.5).astype(np.uint8)

        st.subheader(f"🗓️ Predicted Built-up/Non Built-up Areas for {target_year}")
        st.image(binary_predicted * 255, caption="White = Built-up (1), Black = Non Built-up (0)", use_container_width=True)

        with open(output_tmp_path, "rb") as f:
            st.download_button(
                label="📥 Download Predicted TIFF",
                data=f,
                file_name=f"prediction_{target_year}_{i}.tif",
                mime="image/tiff",
                key=f"download_{i}"
            )
