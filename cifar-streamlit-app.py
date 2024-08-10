from fastai.vision.all import *
import pickle
from pathlib import Path, PosixPath

def run_app():
    # Custom function to replace WindowsPath with PosixPath in unpickling
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'pathlib' and name == 'WindowsPath':
                return PosixPath
            return super().find_class(module, name)

    def load_learner_with_custom_pickle(file, cpu=True):
        map_location = 'cpu' if cpu else default_device()
        with open(file, 'rb') as f:
            return torch.load(f, map_location=map_location, pickle_module=CustomUnpickler)

    # Load the exported learner
    learn = load_learner_with_custom_pickle('cifar_learner.pkl')

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
