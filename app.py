import streamlit as st
from transformers import pipeline, AutoImageProcessor
from PIL import Image
import torch
import os
import zipfile
import sys


KAGGLE_DATASET = "ahmadijaz92/genai-p3-t3" 
MODEL_PATH = "vit-base-food101-results/checkpoint-3552"

def download_model_from_kaggle():
    """
    Checks if the model checkpoint is downloaded. If not, downloads it from Kaggle
    using Streamlit secrets.
    """
    if os.path.exists(MODEL_PATH):
        st.write("Model directory already exists. Skipping download.")
        return

    if 'KAGGLE_USERNAME' not in st.secrets or 'KAGGLE_KEY' not in st.secrets:
        st.error("Kaggle credentials not found in Streamlit secrets.")
        st.error("Please add KAGGLE_USERNAME and KAGGLE_KEY to your app's secrets.")
        st.stop()

    st.info(f"Model not found locally. Downloading from Kaggle dataset: {KAGGLE_DATASET}...")

    # 3. Set up Kaggle API credentials
    os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
        
        st.success(f"Model files downloaded and unzipped (including '{MODEL_PATH}')")
    
    except Exception as e:
        st.error(f"Error downloading from Kaggle: {e}")
        st.stop()

@st.cache_resource
def load_model_pipeline():
    """
    Loads the fine-tuned model and image processor from the local
    directory and wraps them in a pipeline.
    """
    st.write(f"Loading model from local directory: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory not found at {MODEL_PATH}. Please make sure the path is correct after download.")
        return None
    
    try:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

        pipe = pipeline(
            "image-classification",
            model=MODEL_PATH,
            image_processor=image_processor,
            device="cuda" if torch.cuda.is_available() else "cpu" 
        )
        st.write("Model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


st.set_page_config(
    page_title="Food Image Classifier",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food-101 Image Classifier 🍕")
st.write(
    "Upload an image of food, and the Vision Transformer (ViT) model "
    "will predict what it is. This model was fine-tuned on the Food-101 dataset."
)
st.write(f"This app uses a fine-tuned ViT model from Kaggle (`{KAGGLE_DATASET}`).")

download_model_from_kaggle()

classifier = load_model_pipeline()

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and classifier is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("") 
    with st.spinner("Analyzing the image..."):
        try:
            predictions = classifier(image)
            top_pred = predictions[0]
            label = top_pred['label'].replace("_", " ").title()
            score = top_pred['score']
            
            st.success(f"**Prediction: {label}**")
            st.info(f"**Confidence:** {score:.2%}")

            with st.expander("Show Top 5 Predictions"):
                st.dataframe(predictions)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif uploaded_file is None and classifier is not None:
    st.info("Please upload an image file to get started.")

elif classifier is None:
    st.error("Failed to load the model. The app cannot continue.")
    st.stop()
