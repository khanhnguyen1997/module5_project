from fastai.vision.all import *
from pathlib import Path, PosixPath

def run_app():
    # Override WindowsPath to PosixPath if running on a non-Windows system
    def convert_path(filepath):
        return PosixPath(filepath) if os.name != 'nt' else Path(filepath)

    # Load the exported learner, converting the path if necessary
    learn = load_learner(convert_path('cifar_learner.pkl'))

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
