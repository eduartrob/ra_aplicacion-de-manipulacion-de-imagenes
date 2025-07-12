import cv2
import numpy as np

def convert_color(image, color_space="RGB"):
    conversions = {
        "RGB": cv2.COLOR_BGR2RGB,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "GRAYSCALE": cv2.COLOR_BGR2GRAY
    }
    if color_space not in conversions:
        raise ValueError(f"color_space desconocido: {color_space}")
    return cv2.cvtColor(image, conversions[color_space])

def split_channels(image):
    return cv2.split(image)

def merge_channels(ch1, ch2, ch3):
    return cv2.merge([ch1, ch2, ch3])
