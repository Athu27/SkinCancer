import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# Google Drive file ID
file_id = "1w4REf_q9TgU4TPSVmA3nsT1Z4TIAKpy3"  
model_path = "model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):  # Avoid re-downloading if already present
        model_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(model_url)
        
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("Failed to download model. Check the file ID and permissions.")
            return None
    
    return tf.keras.models.load_model(model_path)

# Load model
model = load_model()
if model is None:
    st.stop()  # Stop execution if the model failed to load

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("Skin Cancer Detection")
st.write("Upload an image or capture one using your webcam.")

# Image input options
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
captured_image = st.camera_input("Or capture an image")

# Choose the available image
img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

elif captured_image is not None:
    img = Image.open(captured_image)
    st.image(img, caption="Captured Image", use_column_width=True)

# If an image is provided, make a prediction
if img is not None:
    img_processed = preprocess_image(img)
    prediction = model.predict(img_processed)[0][0]  # Get probability score

    # Display results
    confidence = round(float(prediction) * 100, 2)  # Convert to percentage
    if prediction > 0.83:
        result = f"🔴 Cancer Detected "
    else:
        result = f"🟢 No Cancer Detected"
    
    st.subheader(result)
