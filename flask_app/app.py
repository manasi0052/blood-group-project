from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import logging
from model import model_predict

app = Flask(__name__)

# =============================
# Absolute upload folder (FIX)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'})

    try:
        # Read image
        img = Image.open(file).convert('RGB')
        arr = np.array(img)

        # Predict
        result = model_predict(arr)

        # Save with unique filename (FIX)
        import time
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_name = f"{name}_{int(time.time())}{ext}"

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        img.save(save_path)

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error processing file: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
