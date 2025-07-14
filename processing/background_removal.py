# processing/background_removal.py

import cv2
import numpy as np

# Rango HSV fijo para fondo verde (ajusta según tu fondo)
FIXED_LOWER_HSV = [35, 40, 40]  
FIXED_UPPER_HSV = [85, 255, 255]

# Rango LAB fijo para fondo verde (valores aproximados)
FIXED_LOWER_LAB = [20, 120, 120]
FIXED_UPPER_LAB = [255, 145, 145]

def remove_background_hsv(image):
    lower_hsv = np.array(FIXED_LOWER_HSV, dtype=np.uint8)
    upper_hsv = np.array(FIXED_UPPER_HSV, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

def remove_background_lab(image):
    lower_lab = np.array(FIXED_LOWER_LAB, dtype=np.uint8)
    upper_lab = np.array(FIXED_UPPER_LAB, dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, lower_lab, upper_lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

def grabcut(image, iter_count=5, rect=None):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    if rect is None:
        rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result
