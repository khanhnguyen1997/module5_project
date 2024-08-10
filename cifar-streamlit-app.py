from fastai.vision.all import *
from pathlib import Path, PosixPath, WindowsPath

def load_learner_compatible_with_paths(filepath, cpu=True):
    # Function to map location and handle any path issues
    map_location = 'cpu' if cpu else default_device()

    # Load the object using torch.load
    with open(filepath, 'rb') as f:
        data = torch.load(f, map_location=map_location)
    
    # Recursively replace WindowsPath with PosixPath if needed
    def replace_path(obj):
        if isinstance(obj, WindowsPath):
            return PosixPath(obj)
        elif isinstance(obj, dict):
            return {k: replace_path(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_path(i) for i in obj]
        else:
            return obj

    return replace_path(data)

def run_app():
    # Load the exported learner and fix paths if needed
    learn = load_learner_compatible_with_paths('cifar_learner.pkl')

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
