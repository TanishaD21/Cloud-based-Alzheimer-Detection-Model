import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alzheimer Detection App", layout="centered")

IMG_SIZE = 128
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

@st.cache_resource
def load_trained_model():
    model = load_model("alzheimers_final_model.keras")
    return model

model = load_trained_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    display_img = img.copy()
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return display_img, img_array

def predict_image(img_array):
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100)
    return predicted_class, confidence, predictions

st.title("Alzheimer’s Disease Detection from MRI")
st.write("Upload an MRI scan image to predict the Alzheimer stage.")

st.subheader("Patient Details")
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=1, max_value=120, step=1)
patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    display_img, img_array = preprocess_image(uploaded_file)

    st.subheader("Uploaded MRI Image")
    st.image(display_img, caption="Uploaded MRI", use_container_width=True)

    if st.button("Predict"):
        predicted_class, confidence, predictions = predict_image(img_array)

        st.success(f"Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(CLASS_NAMES, predictions * 100)
        ax.set_ylabel("Probability (%)")
        ax.set_xlabel("Classes")
        ax.set_title("Class-wise Prediction Confidence")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        st.subheader("Detailed Probabilities")
        for class_name, prob in zip(CLASS_NAMES, predictions):
            st.write(f"{class_name}: {prob * 100:.2f}%")

st.warning("Disclaimer: This model is for educational and research purposes only, not for real medical diagnosis.")