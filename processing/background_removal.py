import cv2
import numpy as np

def remove_background_hsv(image, lower_hsv=None, upper_hsv=None):
    """
    Elimina el fondo usando segmentación HSV.
    Args:
        image (np.ndarray): Imagen BGR original.
        lower_hsv (list): Límite inferior del rango HSV (H, S, V).
        upper_hsv (list): Límite superior del rango HSV (H, S, V).
    Returns:
        tuple: (Imagen con fondo negro, Máscara del primer plano)
    """
    if image is None: return None, None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Usar rangos predeterminados si no se proporcionan
    if lower_hsv is None: lower_hsv = [35, 40, 40]
    if upper_hsv is None: upper_hsv = [85, 255, 255]

    lower = np.array(lower_hsv)
    upper = np.array(upper_hsv)
    mask = cv2.inRange(hsv, lower, upper) # Máscara donde el fondo es 255

    # Invertir máscara para conservar objeto (foreground)
    inv_mask = cv2.bitwise_not(mask) # Máscara donde el objeto es 255
    result = cv2.bitwise_and(image, image, mask=inv_mask) # Imagen con fondo negro

    return result, inv_mask


def remove_background_lab(image, lower_lab=None, upper_lab=None):
    """
    Elimina el fondo usando segmentación LAB.
    Args:
        image (np.ndarray): Imagen BGR original.
        lower_lab (list): Límite inferior del rango LAB (L, A, B).
        upper_lab (list): Límite superior del rango LAB (L, A, B).
    Returns:
        tuple: (Imagen con fondo negro, Máscara del primer plano)
    """
    if image is None: return None, None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Usar rangos predeterminados si no se proporcionan
    if lower_lab is None: lower_lab = [20, 120, 120]
    if upper_lab is None: upper_lab = [255, 140, 140]

    lower = np.array(lower_lab)
    upper = np.array(upper_lab)
    mask = cv2.inRange(lab, lower, upper) # Máscara donde el fondo es 255

    # Invertir máscara para conservar objeto
    inv_mask = cv2.bitwise_not(mask) # Máscara donde el objeto es 255
    result = cv2.bitwise_and(image, image, mask=inv_mask) # Imagen con fondo negro

    return result, inv_mask


def grabcut(image, iter_count=5, rect=None):
    """
    Aplica GrabCut para segmentación de fondo.
    Returns:
        tuple: (Imagen con fondo negro, Máscara del primer plano)
    """
    if image is None: return None, None
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    if rect is None:
        # Rectángulo por defecto que cubre la mayor parte de la imagen
        rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"ERROR en GrabCut: {e}. Asegúrate de que la imagen sea válida y el rectángulo tenga sentido.")
        return image, np.zeros(image.shape[:2], dtype=np.uint8) # Retornar original y máscara vacía

    # Crear máscara binaria para foreground (objeto)
    # GC_FGD y GC_PR_FGD son primer plano seguro y probable
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    result = cv2.bitwise_and(image, image, mask=mask2)

    return result, mask2
