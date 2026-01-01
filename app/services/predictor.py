import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from app.config import MODEL_PATH, CLASS_NAMES

# Load model once
model = load_model(MODEL_PATH, safe_mode=False)

def predict_image(image_bytes):
    # Load image (RAW RGB)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img)

    # Binary classification (sigmoid)
    prob = float(prediction[0][0])
    predicted_class = 1 if prob >= 0.5 else 0
    confidence = prob if predicted_class == 1 else 1 - prob

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4)
    }
