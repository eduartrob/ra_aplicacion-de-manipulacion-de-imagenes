import cv2
import numpy as np

def remove_background_hsv(image, lower_hsv, upper_hsv):
    lower_hsv = np.array(lower_hsv, dtype=np.uint8)
    upper_hsv = np.array(upper_hsv, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # Limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Invertir máscara para fondo
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

def remove_background_lab(image, lower_lab, upper_lab):
    lower_lab = np.array(lower_lab, dtype=np.uint8)
    upper_lab = np.array(upper_lab, dtype=np.uint8)
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
    # Crear máscara binaria para el foreground
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result
