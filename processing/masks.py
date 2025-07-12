import cv2
import numpy as np

def apply_threshold(image, thresh_value=128):
    """
    Aplica umbral binario simple.
    image: imagen en escala de grises
    thresh_value: valor de umbral (0-255)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh

def adaptive_threshold(image, max_value=255, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                       threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    Aplica umbral adaptativo.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(image, max_value, method, threshold_type, block_size, C)

def otsu_threshold(image):
    """
    Aplica umbral de Otsu.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def bitwise_and(img1, img2):
    """
    Operación AND bit a bit entre dos imágenes.
    """
    return cv2.bitwise_and(img1, img2)

def bitwise_or(img1, img2):
    """
    Operación OR bit a bit entre dos imágenes.
    """
    return cv2.bitwise_or(img1, img2)

def bitwise_not(img):
    """
    Operación NOT bit a bit sobre una imagen.
    """
    return cv2.bitwise_not(img)
