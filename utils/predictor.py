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
        input_tensor = np.expand_dims(resized, axis=(0, -1))  # Shape: (1, H, W, 1)

    prediction = model.predict(np.expand_dims(input_tensor, 0))[0, ..., 0]
    prediction_resized = cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

    with rasterio.open(input_tiff) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            dst.write(prediction_resized.astype(np.float32), 1)

    return data, prediction_resized

def classify_prediction(prediction_data, output_tiff, reference_tiff, threshold=0.5):
    classified = np.where(prediction_data >= threshold, 1, 0).astype(np.uint8)

    with rasterio.open(reference_tiff) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(classified, 1)

    return classified

def show_images(input_imgs, predicted_imgs, classified_imgs):
    num = len(input_imgs)
    cmap_class = plt.cm.get_cmap('tab10', 2)

    fig, axes = plt.subplots(num, 3, figsize=(15, 5 * num))
    if num == 1:
        axes = [axes]

    for i in range(num):
        axes[i][0].imshow(input_imgs[i], cmap='gray')
        axes[i][0].set_title(f"Input TIFF {i+1}")
        axes[i][0].axis('off')

        axes[i][1].imshow(predicted_imgs[i], cmap='viridis')
        axes[i][1].set_title(f"Predicted Probabilities {i+1}")
        axes[i][1].axis('off')
        fig.colorbar(axes[i][1].images[0], ax=axes[i][1], orientation="vertical")

        im = axes[i][2].imshow(classified_imgs[i], cmap=cmap_class, vmin=0, vmax=1)
        axes[i][2].set_title(f"Binary Classified {i+1}")
        axes[i][2].axis('off')
        cbar = fig.colorbar(im, ax=axes[i][2], ticks=[0, 1])
        cbar.ax.set_yticklabels(['Non Built-up', 'Built-up'])

    plt.tight_layout()
    st.pyplot(fig)