import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from PIL import Image

# Google Drive file ID (extract from shared link)
file_id = "1w4REf_q9TgU4TPSVmA3nsT1Z4TIAKpy3"  # Example: "1a2b3c4d5e6f7g8h9i0j"https://drive.google.com/file/d//view?usp=sharing
model_url = f"https://drive.google.com/uc?id={file_id}"

@st.cache_resource
def load_model():
    response = requests.get(model_url)
    with open("model.h5", "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model("model.h5")

# Load model
model = load_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("Skin Cancer Detection")
st.write("Upload an image to check for skin cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_processed = preprocess_image(img)
    prediction = model.predict(img_processed)

    # Display results
    result = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"
    st.subheader(f"Result: {result}")
