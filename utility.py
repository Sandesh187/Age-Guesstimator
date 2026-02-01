import cv2
import numpy as np

def load_image(path, size=(224, 224)):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Image not found")

    image = cv2.resize(image, size)
    image = image.astype("float32") / 255.0
    return image
