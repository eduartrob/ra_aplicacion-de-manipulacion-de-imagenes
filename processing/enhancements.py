import cv2
import numpy as np

def apply_filter(image, filter_type="median"):
    filters = {
        "blur": lambda img: cv2.blur(img, (5, 5)),
        "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        "bilateral": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        "median": lambda img: cv2.medianBlur(img, 5),
        "sharpen": lambda img: cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])),
        "sobel": lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5),
        "laplacian": lambda img: cv2.Laplacian(img, cv2.CV_64F),
        "canny": lambda img: cv2.Canny(img, 100, 200),
        "emboss": lambda img: cv2.filter2D(img, -1, np.array([[ -2, -1, 0], [ -1, 1, 1], [ 0, 1, 2]])),
        "custom": lambda img: cv2.filter2D(img, -1, np.ones((3, 3), np.float32) / 9)
    }
    func = filters.get(filter_type)
    if func is None:
        print(f"Filtro '{filter_type}' no reconocido, devolviendo imagen original.")
        return image
    return func(image)
