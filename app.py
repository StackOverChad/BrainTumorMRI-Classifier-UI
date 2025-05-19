import streamlit as st # Should be one of the first imports
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess_input
from PIL import Image # This should now work
import numpy as np
import json
import pandas as pd
import os

# --- Call st.set_page_config() as the VERY FIRST Streamlit command ---
st.set_page_config(layout="wide", page_title="Brain Tumor MRI Classifier")

# --- Configuration: Paths to your model and class names file ---
# These paths assume the files are in the SAME directory as this app.py script
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory where app.py is located
MODEL_FILENAME = "brain_tumor_classifier_resnet50v2.keras" # The name of your .keras model file
CLASS_NAMES_FILENAME = "class_names.json" # The name of your class_names.json file

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, CLASS_NAMES_FILENAME)

# --- Parameters (should match your training exactly) ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- Load Model and Class Names ---
# Use st.cache_resource for loading models and other heavy resources to run only once
@st.cache_resource # Caches the loaded model and class names
def load_model_and_classes():
    try:
        # Check if files exist before trying to load
        if not os.path.exists(MODEL_PATH):
            # st.error can be called here as set_page_config is already done
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.error("Please ensure the model file is in the same directory as this script and named correctly.")
            return None, ["Error: Model file missing"] # Return distinct error message
        if not os.path.exists(CLASS_NAMES_PATH):
            st.error(f"Class names file not found at: {CLASS_NAMES_PATH}")
            st.error("Please ensure the class_names.json file is in the same directory and named correctly.")
            return None, ["Error: Class names file missing"] # Return distinct error message

        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print("Model and class names loaded successfully for Streamlit.") # For server-side logging
        return model, class_names
    except Exception as e:
        st.error(f"An error occurred while loading the model or class names: {e}")
        return None, [f"Error loading files: {e}"] # Include exception in error message

model, class_names = load_model_and_classes()

# --- Your Preprocessing and Predict Functions ---
def preprocess_image_for_prediction(image_pil):
    """
    Preprocesses a PIL Image for the model prediction.
    """
    img_resized = image_pil.resize(IMAGE_SIZE, Image.LANCZOS)

    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')

    img_array = np.array(img_resized)
    img_array = tf.cast(img_array, tf.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    img_preprocessed = resnet_v2_preprocess_input(img_batch)
    return img_preprocessed


def predict(image_pil):
    """
    Takes a PIL image, preprocesses it, and returns the predicted class name
    and a dictionary of probabilities for each class.
    """
    # Check if model or class_names failed to load initially
    if model is None or (isinstance(class_names, list) and class_names and "Error" in class_names[0]):
        # The error message would have already been shown by load_model_and_classes
        # We just need to prevent further processing.
        return "Error: Setup incomplete", {"Error": 1.0}

    try:
        preprocessed_image = preprocess_image_for_prediction(image_pil)
        predictions_array = model.predict(preprocessed_image)[0]

        predicted_class_index = np.argmax(predictions_array)
        predicted_class_name = class_names[predicted_class_index]
        
        probabilities = {class_names[i]: float(predictions_array[i]) for i in range(len(class_names))}
        
        return predicted_class_name, probabilities
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error during prediction", {"Error": 1.0}

# --- Streamlit UI Design (Now all st commands are after set_page_config) ---

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main header style */
    .main-header {
        font-size: 2.5em !important; /* Larger font */
        font-weight: bold;
        color: #1E90FF; /* Dodger Blue */
        text-align: center;
        padding-top: 20px;
        padding-bottom: 10px;
    }
    /* Sub-header style */
    .sub-header {
        font-size: 1.2em;
        color: #4A4A4A; /* Dark Gray */
        text-align: center;
        margin-bottom: 30px;
    }
    /* Styling for the file uploader button */
    .stFileUploader > label {
        font-size: 1.1em !important;
    }
    .stFileUploader > div > button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stFileUploader > div > button:hover {
        background-color: #0073e6; /* Darker blue on hover */
    }
    /* Prediction text style */
    .prediction-text {
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
    }
    /* Green for 'no tumor', Red for tumor types */
    .no-tumor { color: #28A745; } /* Green */
    .tumor-detected { color: #DC3545; } /* Red */
    </style>
""", unsafe_allow_html=True)

# --- Page Title and Description ---
st.markdown("<p class='main-header'>🧠 Brain Tumor MRI Classification 🔬</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Upload a brain MRI scan to classify it. "
    "The model can detect Glioma, Meningioma, Pituitary tumors, or No Tumor.</p>",
    unsafe_allow_html=True
)

# --- Sidebar for Information ---
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a Deep Learning model (ResNet50V2 architecture) "
    "to classify Brain MRI images. "
    "It was trained on the 'Brain Tumor MRI Dataset' from Kaggle."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How to Use:")
st.sidebar.markdown("1. Click on **'Browse files'** or drag and drop an MRI image into the designated area.")
st.sidebar.markdown("2. The model will process the image.")
st.sidebar.markdown("3. The predicted class and confidence scores will be displayed.")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Disclaimer:** This tool is for educational and demonstration purposes only. "
                   "It should **NOT** be used for actual medical diagnosis. Always consult a qualified medical professional.")

# --- Main Content Area for Image Upload and Prediction ---

# Check if model and class names loaded correctly before proceeding with UI that needs them
if model is None or (isinstance(class_names, list) and class_names and "Error" in class_names[0]):
    st.error("Application setup failed: Model or class names could not be loaded. Please check the terminal/server logs for more details.")
else:
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a brain MRI scan."
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)

        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(image_pil, caption="Uploaded MRI Scan", use_column_width='always')

        with col2:
            with st.spinner("🧠 Analyzing the scan... Please wait."):
                predicted_class_name, probabilities = predict(image_pil)

            if "Error" not in predicted_class_name : # Check for error string
                prediction_color_class = "no-tumor" if "notumor" in predicted_class_name.lower() else "tumor-detected"
                
                st.markdown(
                    f"<p class='prediction-text {prediction_color_class}'>Predicted Class: {predicted_class_name}</p>",
                    unsafe_allow_html=True
                )
                
                st.subheader("Confidence Scores:")
                probs_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
                probs_df = probs_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
                
                st.table(probs_df.style.format({"Probability": "{:.2%}"})
                                     .highlight_max(subset=['Probability'], color='lightgreen', axis=0)
                                     .set_properties(**{'width': '100px'}))

                st.subheader("Probabilities Chart:")
                chart_data = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
                st.bar_chart(chart_data, height=250)

                if "notumor" in predicted_class_name.lower():
                    st.success("The model suggests no tumor is present in this scan.")
                else:
                    st.warning(f"The model suggests a {predicted_class_name} might be present. "
                               "This is not a diagnosis. Please consult a medical expert.")
            # No explicit else here for prediction error, as predict() function already calls st.error

    else:
        st.info("👈 Upload an image using the uploader above to see a prediction.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:grey; font-size:0.9em;'>"
            "Brain Tumor MRI Classifier Demo | Model: ResNet50V2"
            "</p>", unsafe_allow_html=True)