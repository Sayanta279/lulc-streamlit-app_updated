
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
import streamlit as st

def predict_tiff(input_tiff, output_tiff, model, target_shape=(256, 256)):
    with rasterio.open(input_tiff) as src:
        data = src.read(1)
        original_shape = data.shape
        resized = cv2.resize(data, target_shape, interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized, axis=(0, -1))

    prediction = model.predict(input_tensor)[0, ..., 0]
    prediction_resized = cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

    with rasterio.open(input_tiff) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            dst.write(prediction_resized.astype(rasterio.float32), 1)

    return data, prediction_resized

def classify_prediction(prediction_data, output_tiff, reference_tiff, thresholds=[0.2, 0.4, 0.6, 0.8], class_values=[1, 2, 3, 4, 5]):
    classified = np.zeros_like(prediction_data, dtype=np.int32)
    for i, thresh in enumerate(thresholds):
        if i == 0:
            classified[prediction_data <= thresh] = class_values[i]
        else:
            classified[(prediction_data > thresholds[i-1]) & (prediction_data <= thresh)] = class_values[i]
    classified[prediction_data > thresholds[-1]] = class_values[-1]

    with rasterio.open(reference_tiff) as src:
        profile = src.profile
    profile.update(dtype=rasterio.int32, count=1)
    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(classified, 1)

    return classified

def show_images(input_imgs, predicted_imgs, classified_imgs):
    num = len(input_imgs)
    fig, axs = plt.subplots(num, 3, figsize=(18, 6 * num))

    if num == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(num):
        axs[i, 0].imshow(input_imgs[i], cmap='gray')
        axs[i, 0].set_title(f'Input {i+1}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(predicted_imgs[i], cmap='viridis')
        axs[i, 1].set_title(f'Predicted {i+1}')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(classified_imgs[i], cmap='tab10')
        axs[i, 2].set_title(f'Classified {i+1}')
        axs[i, 2].axis('off')

    st.pyplot(fig)
