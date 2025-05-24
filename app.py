# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Load model
model = load_model("car_brand_classifier_efficientnet.h5")

# Class labels (update if you have more or different classes)
class_labels = sorted([
    'audi', 'bmw', 'chevrolet', 'ford', 'honda',
    'hyundai', 'kia', 'mazda', 'mercedes', 'toyota'
])

# Title
st.title("Car Brand Classifier 🚗")

# Upload image
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Predict
    predictions = model.predict(img_preprocessed)
    top_class = np.argmax(predictions[0])
    confidence = predictions[0][top_class]

    st.subheader(f"Prediction: {class_labels[top_class]}")
    st.write(f"Confidence: {confidence:.2f}")
