# model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Point directly at your .h5
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/final_cnn.h5')
_model = None

def load_model_once():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

# These labels must match your training order
class_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

def preprocess_blood_image(img_array):
    """
    img_array: H×W×3 uint8 RGB (0–255)
    Returns: 64×64×3 float32 (0–1)
    """
    # 1) Resize
    img = Image.fromarray(img_array).resize((64, 64))
    # 2) Convert to array and normalize
    arr = np.array(img).astype('float32') / 255.0
    return arr  # shape (64,64,3)

def model_predict(image_array):
    """
    image_array: any size H×W×3 uint8 or float [0–1]
    """
    model = load_model_once()
    # 1) Ensure it’s uint8→RGB
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype('uint8')
    # 2) Preprocess
    proc = preprocess_blood_image(image_array)
    # 3) Batch and predict
    batch = np.expand_dims(proc, axis=0)
    preds = model.predict(batch)[0]
    idx = np.argmax(preds)
    return {
        'blood_type': class_labels[idx],
        'confidence': round(float(preds[idx]) * 100, 2)
    }

