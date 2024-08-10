from pathlib import PosixPath, WindowsPath
import torch
import pickle

def map_windows_path_to_posix(obj):
    if isinstance(obj, dict):
        return {k: map_windows_path_to_posix(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [map_windows_path_to_posix(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(map_windows_path_to_posix(i) for i in obj)
    elif isinstance(obj, WindowsPath):
        return PosixPath(str(obj))
    return obj

def load_learner_compatible(filepath):
    # Load the model file ensuring it works for Linux systems
    with open(filepath, 'rb') as f:
        learner = torch.load(f, map_location=torch.device('cpu'))
    
    # Recursively replace WindowsPath with PosixPath
    learner = map_windows_path_to_posix(learner)
    
    return learner

def run_app():
    # Load the exported learner and fix paths if needed
    learn = load_learner_compatible('cifar_learner.pkl')
    
    # Streamlit app title
    st.title("CIFAR Image Classifier")
    
    # Example of using the loaded learner
    img = load_image("path_to_image.png")
    pred_idx, probs = learn.predict(img)
    st.write(f"Prediction: {pred_idx}")
    st.write(f"Probability: {probs[pred_idx]:.4f}")

if __name__ == '__main__':
    run_app()
