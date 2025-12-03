import numpy as np
from PIL import Image

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)
