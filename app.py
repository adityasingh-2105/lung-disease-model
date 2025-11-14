"""
Streamlit web application for chest X-ray classification.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import create_model
from preprocessing import preprocess_uploaded_image, get_val_transforms
import os


# Page configuration
st.set_page_config(
    page_title="Lung Disease Classifier",
    page_icon="🫁",
    layout="wide"
)

# Model configuration
MODEL_PATH = "models/best_model.pth"
NUM_CLASSES = 2
INPUT_CHANNELS = 1
CLASS_NAMES = ["Normal", "Pneumonia"]


@st.cache_resource
def load_model():
    """
    Load the trained model (cached for performance).
    
    Returns:
        Loaded and ready-to-use model
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None
    
    try:
        model = create_model(
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS,
            pretrained_path=MODEL_PATH
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def predict(model, image_tensor):
    """
    Make prediction on preprocessed image.
    
    Args:
        model: Trained CNN model
        image_tensor: Preprocessed image tensor
        
    Returns:
        Predicted class probabilities and class name
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = float(confidence.item() * 100)  # Convert to Python float
        
        # Get probabilities for all classes and convert to Python floats
        all_probabilities = probabilities[0].cpu().numpy()
        all_probabilities = [float(p) for p in all_probabilities]  # Convert numpy array to list of Python floats
    
    return predicted_class, confidence_score, all_probabilities


def main():
    """
    Main Streamlit application.
    """
    # Title and description
    st.title("🫁 Lung Disease Classifier")
    st.markdown("""
    Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**.
    
    This application uses a lightweight Convolutional Neural Network (CNN) trained 
    on chest X-ray images to assist in detecting lung diseases.
    
    **Note:** This is a demonstration tool and should not be used as a substitute 
    for professional medical diagnosis.
    """)
    
    st.divider()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Preprocess image
                with st.spinner("Processing image..."):
                    image_tensor = preprocess_uploaded_image(uploaded_file)
                
                # Make prediction
                with st.spinner("Analyzing..."):
                    predicted_class, confidence, all_probabilities = predict(model, image_tensor)
                
                # Display results
                st.markdown("### Prediction")
                
                # Color code based on prediction
                if predicted_class == "Normal":
                    st.success(f"**{predicted_class}** (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"**{predicted_class}** (Confidence: {confidence:.2f}%)")
                
                st.markdown("### Confidence Scores")
                
                # Progress bars for each class
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = all_probabilities[i] * 100  # Already converted to Python float
                    prob_value = prob / 100  # Ensure value is between 0 and 1 for progress bar
                    
                    st.progress(prob_value, text=f"{class_name}: {prob:.2f}%")
                
                # Additional info
                st.markdown("---")
                st.markdown("### ℹ️ Information")
                st.info(
                    "**Disclaimer:** This tool is for educational and demonstration "
                    "purposes only. It should not replace professional medical advice, "
                    "diagnosis, or treatment. Always consult with qualified healthcare "
                    "professionals for medical concerns."
                )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please ensure you uploaded a valid chest X-ray image.")
        else:
            st.info("👈 Please upload an X-ray image to get started.")
            st.markdown("""
            **How to use:**
            1. Click "Browse files" or drag and drop an image
            2. Wait for the model to process the image
            3. Review the prediction and confidence scores
            """)


if __name__ == '__main__':
    main()

