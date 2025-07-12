import cv2
import numpy as np

def to_bgr(image):
    """
    Convierte una imagen en escala de grises (2D) a BGR (3 canales).
    Si la imagen ya tiene 3 canales, la devuelve sin cambios.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image
    else:
        raise ValueError("La imagen tiene un formato no soportado")
