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
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# App Title
st.set_page_config(layout="wide")
st.title("🌍 Land Use Land Cover (LULC) Prediction for Target Year")

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

# Authenticate Google Drive
@st.cache_resource
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

drive = authenticate_drive()

# File uploader and year selector
uploaded_files = st.file_uploader("📂 Upload at least 2 Binary LULC GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
target_year = st.number_input("📅 Enter the year you want to predict", min_value=1900, max_value=2100, step=1)

def extract_year(filename):
    try:
        return int(os.path.basename(filename.name).split('_')[-1].split('.')[0])
    except:
        return 0

if uploaded_files and len(uploaded_files) >= 2 and target_year:
    st.info("⏳ Processing files and predicting for year " + str(target_year))

    uploaded_files.sort(key=lambda f: abs(extract_year(f) - target_year))
    selected_files = uploaded_files[:2]

    input_images = []
    predicted_images = []
    classified_images = []

    # Use only one output for the target year
    with NamedTemporaryFile(delete=False, suffix=".tif") as temp_input:
        temp_input.write(selected_files[0].read())
        input_path = temp_input.name

    predicted_path = input_path.replace(".tif", f"_predicted_{target_year}.tif")
    classified_path = input_path.replace(".tif", f"_classified_{target_year}.tif")

    input_img, pred_img = predict_tiff(input_path, predicted_path, model)
    class_img = classify_prediction(pred_img, classified_path, input_path)

    input_images.append(input_img)
    predicted_images.append(pred_img)
    classified_images.append(class_img)

    st.success("✅ Prediction and Classification Completed")
    show_images(input_images, predicted_images, classified_images)

    # Upload to Google Drive and generate download link
    gfile = drive.CreateFile({'title': f"predicted_{target_year}.tif"})
    gfile.SetContentFile(predicted_path)
    gfile.Upload()
    download_url = f"https://drive.google.com/uc?id={gfile['id']}&export=download"

    st.markdown(f"📥 [Download Predicted TIFF for {target_year}]({download_url})")
else:
    st.warning("⚠️ Please upload at least two TIFF files and select a target year.")