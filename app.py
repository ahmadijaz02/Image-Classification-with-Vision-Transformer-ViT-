import streamlit as st
from PIL import Image
from transformers import pipeline, AutoImageProcessor
import os
import torch  # Make sure torch is imported

# --- Page Configuration ---
st.set_page_config(
    page_title="Food Image Classifier",
    page_icon="🍔",
    layout="centered"
)

# --- Model Loading ---

# -----------------------------------------------------------------
# --- THIS IS THE UPDATED PATH ---
# -----------------------------------------------------------------
# We point directly to the checkpoint folder you downloaded.
# Use the *last* checkpoint, as it's the most trained.
MODEL_PATH = "./vit-base-food101-results/checkpoint-3552"
# -----------------------------------------------------------------


@st.cache_resource
def load_model_pipeline():
    """
    Loads the fine-tuned model and image processor from the local
    directory and wraps them in a pipeline.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory not found at {MODEL_PATH}. Please make sure the path is correct.")
        return None
    
    try:
        # Load the processor
        image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        
        # Load the pipeline
        pipe = pipeline(
            "image-classification",
            model=MODEL_PATH,
            image_processor=image_processor,
            device="cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pipeline
classifier = load_model_pipeline()

# --- App Title and Description ---
st.title("🍔 Food-101 Image Classifier 🍕")
st.write(
    "Upload an image of food, and the Vision Transformer (ViT) model "
    "will predict what it is. This model was fine-tuned on the Food-101 dataset."
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and classifier is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # --- Classification ---
    st.write("") # Add a little space
    with st.spinner("Analyzing the image..."):
        try:
            # Get predictions
            predictions = classifier(image)
            
            # Get the top prediction
            top_pred = predictions[0]
            label = top_pred['label'].replace("_", " ").title()
            score = top_pred['score']
            
            # Display the result
            st.success(f"**Prediction: {label}**")
            st.info(f"**Confidence:** {score:.2%}")
            
            # (Optional) Show top 5 predictions in an expandable section
            with st.expander("Show Top 5 Predictions"):
                st.dataframe(predictions)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif uploaded_file is None:
    st.info("Please upload an image file to get started.")