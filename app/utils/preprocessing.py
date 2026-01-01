from PIL import Image
import numpy as np
from io import BytesIO
from app.config import IMAGE_SIZE

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array