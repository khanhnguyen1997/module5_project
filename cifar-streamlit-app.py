from fastai.vision.all import *
from pathlib import Path, PosixPath, WindowsPath
import streamlit as st
import torch

def fix_paths(obj):
    if isinstance(obj, WindowsPath):
        return PosixPath(obj)
    elif isinstance(obj, dict):
        return {k: fix_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_paths(i) for i in obj]
    else:
        return obj

def load_learner_compatible(filepath, cpu=True):
    map_location = 'cpu' if cpu else default_device()
    
    # Load the model
    with open(filepath, 'rb') as f:
        learner = torch.load(f, map_location=map_location)
    
    # Fix any WindowsPath issues
    learner = fix_paths(learner)
    
    return learner

def run_app():
    # Load the exported learner and fix paths if needed
    learn = load_learner_compatible('cifar_learner.pkl')

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
