import streamlit as st
from fastai.vision.all import *

def run_app():
    # Load the exported learner
    learn = load_learner(r'D:\Northeastern\EAI6010\cifar_learner.pkl')

    # Streamlit app title
    st.title("CIFAR Image Classifier")

    # File uploader allows users to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = PILImage.create(uploaded_file)
        st.image(img.to_thumb(256, 256), caption='Uploaded Image', use_column_width=True)

        # Make prediction
        pred_class, pred_idx, probs = learn.predict(img)
        st.write(f"Prediction: {pred_class}")
        st.write(f"Probability: {probs[pred_idx]:.4f}")

if __name__ == '__main__':
    run_app()