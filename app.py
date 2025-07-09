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
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

st.set_page_config(layout="wide")
st.title("🌍 LULC Prediction for Target Year with Google Drive API")

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

@st.cache_resource
def init_drive_service():
    service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return build('drive', 'v3', credentials=creds)

drive_service = init_drive_service()

uploaded_files = st.file_uploader("📂 Upload at least 2 Binary LULC GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
target_year = st.number_input("📅 Enter the year you want to predict", min_value=1900, max_value=2100, step=1)

def extract_year(filename):
    try:
        return int(os.path.basename(filename.name).split('_')[-1].split('.')[0])
    except:
        return 0

def upload_to_drive(file_path, filename):
    file_metadata = {'name': filename}
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return f"https://drive.google.com/uc?id={uploaded_file['id']}&export=download"

if uploaded_files and len(uploaded_files) >= 2:
    st.info("⏳ Processing files and predicting for year " + str(target_year))

    uploaded_files.sort(key=lambda f: abs(extract_year(f) - target_year))
    selected_files = uploaded_files[:2]

    input_images = []
    predicted_images = []
    classified_images = []

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

    download_url = upload_to_drive(predicted_path, f"predicted_{target_year}.tif")
    st.markdown(f"📥 [Download Predicted TIFF for {target_year}]({download_url})")
else:
    st.warning("⚠️ Please upload at least two TIFF files and select a target year.")