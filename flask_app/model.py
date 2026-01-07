# model.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Path to the trained model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'final_cnn.h5')
_model = None

# Blood group labels must match training class indices order
class_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def load_model_once():
    """
    Load the Keras model only once for reuse.
    """
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    return _model


def preprocess_blood_image(img_array):
    """
    img_array: H×W×3 or H×W×1 (uint8 or float32)
    Returns: 224×224×3 float32 (0–1)
    """
    # Convert grayscale to RGB if needed
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    # Resize and normalize
    img = Image.fromarray(img_array).resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    return arr  # shape: 224×224×3

def model_predict(image_array):
    model = load_model_once()

    # Handle input types
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype('uint8')

    proc = preprocess_blood_image(image_array)
    batch = np.expand_dims(proc, axis=0)
    preds = model.predict(batch)[0]
    idx = np.argmax(preds)
    return {
        'blood_type': class_labels[idx],
        'confidence': round(float(preds[idx]) * 100, 2)
    }