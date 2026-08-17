import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


# ---------------------------------------
# Configuration
# ---------------------------------------

IMAGE_SIZE = (224, 224)

MODEL_PATH = "models/class_weighted_wildlife_model.keras"

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------
# Load Model
# ---------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Wildlife model loaded successfully!")


# ---------------------------------------
# Load Class Names
# ---------------------------------------

TEST_DIR = "data/processed/test"

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.class_names

print("Number of classes:", len(class_names))


# ---------------------------------------
# Home Route
# ---------------------------------------

@app.route("/")
def home():

    return jsonify({
        "message": "Wildlife Species Recognition API is running"
    })


# ---------------------------------------
# Prediction Route
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image was uploaded
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["image"]


    # Check filename
    if file.filename == "":

        return jsonify({
            "error": "No image selected"
        }), 400


    # Save uploaded image
    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(image_path)


    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )


    # Convert image to array
    image_array = tf.keras.utils.img_to_array(
        image
    )


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )


    # Top 3 predictions
    top_3_indices = np.argsort(
        predictions[0]
    )[-3:][::-1]


    top_3 = []


    for index in top_3_indices:

        species = class_names[index]

        confidence = (
            predictions[0][index] * 100
        )

        top_3.append({
            "species": species,
            "confidence": round(
                float(confidence),
                2
            )
        })


    # Best prediction
    best_confidence = top_3[0]["confidence"]


    # Confidence status
    if best_confidence >= 70:

        status = "High Confidence"

    elif best_confidence >= 40:

        status = "Medium Confidence"

    else:

        status = "Low Confidence"


    return jsonify({

        "prediction": top_3[0]["species"],

        "confidence": best_confidence,

        "status": status,

        "top_3": top_3

    })


# ---------------------------------------
# Run Flask
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )