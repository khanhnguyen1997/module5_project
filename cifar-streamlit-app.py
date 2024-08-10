import torch
from pathlib import Path  # Used for handling file paths
import streamlit as st  # Assuming Streamlit is being used for the app

def load_learner_compatible(filepath):
    # Load the model file ensuring it works for Linux systems
    learner = torch.load(filepath, map_location=torch.device('cpu'))
    return learner

def run_app():
    # Load the exported learner and fix paths if needed
    learn = load_learner_compatible('cifar_learner_linux.pkl')
    
    # Streamlit app title
    st.title("CIFAR Image Classifier")
    
    # Example of using the loaded learner
    img = load_image("path_to_image.png")  # Replace with actual image loading method
    pred_idx, probs = learn.predict(img)
    st.write(f"Prediction: {pred_idx}")
    st.write(f"Probability: {probs[pred_idx]:.4f}")

if __name__ == '__main__':
    run_app()
