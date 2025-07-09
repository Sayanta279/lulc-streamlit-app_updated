# LULC Predictor Web App

A Streamlit-based web application to perform Land Use Land Cover (LULC) prediction using a trained U-Net model.

## Features

- Upload one or more `.tif` raster images
- Predict LULC using a trained U-Net model
- Classify predicted output into discrete classes
- Visualize and download the results

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload files to a public GitHub repository.
2. Go to https://streamlit.io/cloud
3. Connect your GitHub and deploy the app.