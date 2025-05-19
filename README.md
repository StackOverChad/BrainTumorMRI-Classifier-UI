# Brain Tumor MRI Classifier UI

A Streamlit web application for classifying brain tumor MRI scans using a deep learning model (ResNet50V2).

## Features
- Upload MRI images (JPG, JPEG, PNG).
- Classifies images into: Glioma, Meningioma, Pituitary, or No Tumor.
- Displays prediction probabilities.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/StackOverChad/BrainTumorMRI-Classifier-UI.git
    cd BrainTumorMRI-Classifier-UI
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If users encounter SSL issues, they might need to use `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`)*

4.  **Ensure model files are present:**
    This repository includes `brain_tumor_classifier_resnet50v2.keras` and `class_names.json`.

## Running the Application
```bash
streamlit run app.py