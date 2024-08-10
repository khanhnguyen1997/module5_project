from pathlib import PosixPath, WindowsPath
import torch

def load_learner_compatible(filepath):
    # Load the model file ensuring it works for Linux systems
    with open(filepath, 'rb') as f:
        learner = torch.load(f, map_location=torch.device('cpu'))
    return learner

def run_app():
    # Load the exported learner and fix paths if needed
    learn = load_learner_compatible(r'cifar_learner_linux.pkl')
    
    # Streamlit app title
    st.title("CIFAR Image Classifier")
    
    # Example of using the loaded learner
    img = load_image("path_to_image.png")
    pred_idx, probs = learn.predict(img)
    st.write(f"Prediction: {pred_idx}")
    st.write(f"Probability: {probs[pred_idx]:.4f}")

if __name__ == '__main__':
    run_app()
