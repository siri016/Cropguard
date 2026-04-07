from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# ✅ Load trained model
model = load_model("model/crop_model.h5")

# ✅ Class names (must match dataset order)
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# ✅ Home route
@app.route('/')
def home():
    return "🌿 CropGuard API is Running!"

# ✅ UI Page
@app.route('/ui')
def ui():
    return render_template("index.html")

# ✅ Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        # Read and preprocess image
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = {
            "disease": class_names[class_index],
            "confidence": round(confidence, 4)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)
