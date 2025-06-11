from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load your model once when the app starts
model = tf.keras.models.load_model("model.h5")

# Replace with your actual class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((96, 96))  # Make sure this matches your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return "ML Model API is Live"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img_array = prepare_image(file.read())
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        return jsonify({"prediction": pred_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
